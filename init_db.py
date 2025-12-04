import sqlite3
import hashlib
import os

DB = "app.db"

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def init_db():
    if os.path.exists(DB):
        print("Database already exists:", DB)
    conn = sqlite3.connect(DB)
    cur = conn.cursor()


    cur.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'user'
        )
    ''')

    cur.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            age INTEGER,
            gender INTEGER,
            reg_time INTEGER,
            distance REAL,
            event_type INTEGER,
            past_att INTEGER,
            reminder INTEGER,
            ticket INTEGER,
            weekend INTEGER,
            predicted INTEGER,
            probability REAL
        )
    ''')


    admin_user = "admin"
    admin_pass = "admin123"
    cur.execute("SELECT * FROM users WHERE username=?", (admin_user,))
    if cur.fetchone() is None:
        cur.execute(
            "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
            (admin_user, hash_password(admin_pass), "admin")
        )
        print(f"Created admin user -> username: {admin_user}, password: {admin_pass}")

    conn.commit()
    conn.close()
    print("DB initialized:", DB)

if __name__ == "__main__":
    init_db()
