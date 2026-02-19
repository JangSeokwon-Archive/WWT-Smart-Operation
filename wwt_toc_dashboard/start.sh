#!/bin/bash

echo "🌊 WWT TOC Dashboard - Premium Edition"
echo "======================================"
echo ""
echo "📦 패키지 설치 중..."
pip install -r requirements.txt -q

echo ""
echo "✅ 설치 완료!"
echo ""
echo "🚀 대시보드 실행 중..."
echo ""
echo "📌 브라우저에서 다음 주소로 접속하세요:"
echo "   http://localhost:8501"
echo ""
echo "💡 종료하려면 Ctrl+C를 누르세요"
echo ""

streamlit run streamlit_app.py
