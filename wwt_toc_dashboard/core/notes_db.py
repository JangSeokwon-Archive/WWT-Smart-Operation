import os
import json
import sqlite3
from typing import List, Optional

DB_DIR = "data_notes"
DB_PATH = os.path.join(DB_DIR, "notes.sqlite")
MEDIA_DIR = os.path.join(DB_DIR, "media")

def init_db():
    os.makedirs(DB_DIR, exist_ok=True)
    os.makedirs(MEDIA_DIR, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS notes (
      note_id TEXT PRIMARY KEY,
      created_at TEXT,
      final_toc REAL,
      flow REAL,
      temp REAL,
      risk TEXT,
      memo TEXT,
      media_json TEXT
    )
    """)
    con.commit()
    con.close()

def insert_note(note_id: str, created_at: str, final_toc, flow, temp,
                risk: str, memo: str, media_paths: List[str]):
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
      INSERT INTO notes(note_id, created_at, final_toc, flow, temp, risk, memo, media_json)
      VALUES(?,?,?,?,?,?,?,?)
    """, (note_id, created_at, final_toc, flow, temp, risk, memo, json.dumps(media_paths)))
    con.commit()
    con.close()

def list_notes(limit: int = 200):
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
      SELECT note_id, created_at, final_toc, flow, temp, risk, memo, media_json
      FROM notes
      ORDER BY created_at DESC
      LIMIT ?
    """, (limit,))
    rows = cur.fetchall()
    con.close()
    return rows

def get_note(note_id: str):
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
      SELECT note_id, created_at, final_toc, flow, temp, risk, memo, media_json
      FROM notes
      WHERE note_id = ?
    """, (note_id,))
    row = cur.fetchone()
    con.close()
    return row

def update_note_memo(note_id: str, memo: str, media_paths: Optional[List[str]] = None) -> int:
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    if media_paths is None:
        cur.execute("UPDATE notes SET memo = ? WHERE note_id = ?", (memo, note_id))
    else:
        cur.execute(
            "UPDATE notes SET memo = ?, media_json = ? WHERE note_id = ?",
            (memo, json.dumps(media_paths), note_id),
        )
    changed = cur.rowcount if cur.rowcount is not None else 0
    con.commit()
    con.close()
    return changed

def delete_notes(note_ids: List[str]) -> int:
    if not note_ids:
        return 0
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    placeholders = ",".join(["?"] * len(note_ids))
    cur.execute(f"DELETE FROM notes WHERE note_id IN ({placeholders})", tuple(note_ids))
    deleted = cur.rowcount if cur.rowcount is not None else 0
    con.commit()
    con.close()
    return deleted

def save_media(file_name: str, data: bytes) -> str:
    safe = file_name.replace("/", "_").replace("\\", "_")
    path = os.path.join(MEDIA_DIR, safe)
    with open(path, "wb") as f:
        f.write(data)
    return path
