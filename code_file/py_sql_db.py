import pandas as pd
import psycopg

df = pd.read_csv("/Users/d.h/dev/ai_chatbot/data_file/netflix_titles.csv")

# [{"show_id": "s1", "description": "..."}, {...}, ...] 형태로 변환
rows = df[["show_id", "description"]].to_dict(orient="records")

with psycopg.connect(
    host="localhost", port=5434,
    user="postgres", password="admin1234", dbname="chatbot"
) as conn:
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS netflix_desc (
                show_id     TEXT PRIMARY KEY,
                description TEXT
            )
        """)
        cur.executemany(
            "INSERT INTO netflix_desc (show_id, description) "
            "VALUES (%(show_id)s, %(description)s) "
            "ON CONFLICT (show_id) DO NOTHING",
            rows,
        )
    # with 블록이 끝나면 자동으로 commit