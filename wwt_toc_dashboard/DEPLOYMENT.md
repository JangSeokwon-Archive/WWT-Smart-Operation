# 🚀 배포 가이드

## 로컬 실행

### macOS / Linux
```bash
chmod +x start.sh
./start.sh
```

### Windows
```cmd
start.bat
```

### 수동 실행
```bash
# 1. 패키지 설치
pip install -r requirements.txt

# 2. 실행
streamlit run streamlit_app.py
```

---

## 클라우드 배포

### Streamlit Cloud (무료)

1. **GitHub 리포지토리 생성**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin YOUR_REPO_URL
   git push -u origin main
   ```

2. **Streamlit Cloud 접속**
   - https://share.streamlit.io 방문
   - GitHub 계정으로 로그인
   - "New app" 클릭
   - 리포지토리 선택
   - Main file: `streamlit_app.py`
   - Deploy 클릭

3. **완료!**
   - 자동으로 URL 생성됨 (예: https://yourapp.streamlit.app)

### Heroku

1. **Procfile 생성**
   ```
   web: sh setup.sh && streamlit run streamlit_app.py
   ```

2. **setup.sh 생성**
   ```bash
   mkdir -p ~/.streamlit/
   echo "\
   [server]\n\
   headless = true\n\
   port = $PORT\n\
   enableCORS = false\n\
   \n\
   " > ~/.streamlit/config.toml
   ```

3. **배포**
   ```bash
   heroku create your-app-name
   git push heroku main
   ```

### Docker

1. **Dockerfile 생성**
   ```dockerfile
   FROM python:3.9-slim
   
   WORKDIR /app
   
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   
   COPY . .
   
   EXPOSE 8501
   
   CMD ["streamlit", "run", "streamlit_app.py"]
   ```

2. **빌드 및 실행**
   ```bash
   docker build -t wwt-dashboard .
   docker run -p 8501:8501 wwt-dashboard
   ```

### AWS EC2

1. **EC2 인스턴스 생성** (Ubuntu 22.04)

2. **SSH 접속 및 설치**
   ```bash
   sudo apt update
   sudo apt install python3-pip -y
   git clone YOUR_REPO_URL
   cd wwt_dashboard_redesigned
   pip3 install -r requirements.txt
   ```

3. **백그라운드 실행**
   ```bash
   nohup streamlit run streamlit_app.py &
   ```

4. **도메인 연결** (선택사항)
   - Route 53에서 도메인 설정
   - Nginx 리버스 프록시 설정

---

## 환경 변수 설정

프로덕션 환경에서는 `.streamlit/secrets.toml` 사용:

```toml
# .streamlit/secrets.toml
[database]
host = "your-db-host"
port = 5432
user = "your-user"
password = "your-password"

[api]
key = "your-api-key"
```

코드에서 사용:
```python
import streamlit as st

db_host = st.secrets["database"]["host"]
api_key = st.secrets["api"]["key"]
```

---

## 성능 최적화

### 캐싱 전략
```python
# TTL 설정으로 주기적 갱신
@st.cache_data(ttl=300)  # 5분
def load_data():
    return pd.read_csv('data.csv')

# 리소스 캐싱 (DB 연결 등)
@st.cache_resource
def init_connection():
    return database.connect()
```

### 대용량 데이터
```python
# 페이징 처리
@st.cache_data
def load_page(page, page_size=100):
    start = page * page_size
    end = start + page_size
    return df.iloc[start:end]

# 데이터 압축
df.to_parquet('data.parquet', compression='gzip')
df = pd.read_parquet('data.parquet')
```

---

## 보안

### 1. 인증 추가
```python
import streamlit_authenticator as stauth

# 사용자 정보
names = ['John Doe', 'Jane Smith']
usernames = ['jdoe', 'jsmith']
passwords = ['xxx', 'yyy']

authenticator = stauth.Authenticate(
    names, usernames, passwords,
    'cookie_name', 'signature_key', cookie_expiry_days=30
)

name, authentication_status, username = authenticator.login('Login', 'main')

if authentication_status:
    st.write(f'Welcome *{name}*')
    # 대시보드 코드...
else:
    st.error('Username/password is incorrect')
```

### 2. HTTPS 설정
```toml
# .streamlit/config.toml
[server]
enableXsrfProtection = true
enableCORS = false
```

### 3. 민감 정보 보호
- `.gitignore`에 `.streamlit/secrets.toml` 추가
- 환경 변수 사용
- API 키 암호화

---

## 모니터링

### Google Analytics 연동
```python
# streamlit_app.py
st.markdown("""
<script async src="https://www.googletagmanager.com/gtag/js?id=GA_MEASUREMENT_ID"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'GA_MEASUREMENT_ID');
</script>
""", unsafe_allow_html=True)
```

### 로그 설정
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
logger.info("Dashboard started")
```

---

## 문제 해결

### 메모리 부족
```bash
# Streamlit 메모리 제한 증가
streamlit run streamlit_app.py --server.maxUploadSize 1000
```

### 포트 변경
```bash
streamlit run streamlit_app.py --server.port 8502
```

### 디버그 모드
```bash
streamlit run streamlit_app.py --logger.level=debug
```

---

## 추가 리소스

- [Streamlit 공식 문서](https://docs.streamlit.io)
- [Streamlit Community](https://discuss.streamlit.io)
- [Deployment Guide](https://docs.streamlit.io/streamlit-community-cloud)
