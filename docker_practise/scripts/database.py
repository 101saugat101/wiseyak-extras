import psycopg2
from psycopg2.extras import RealDictCursor
from fastapi import HTTPException

# Database connection details
DB_CONFIG = {
    "dbname": "similarity_search",
    "user": "postgres",
    "password": "heheboii420",
    "host": "localhost",
    "port": 5432
}

# Function to get database connection
def get_db_connection():
    try:
        conn = psycopg2.connect(**DB_CONFIG, cursor_factory=RealDictCursor)
        return conn
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database connection error: {e}")

# Function to create the necessary table
def create_table():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE EXTENSION IF NOT EXISTS vector;
        CREATE TABLE IF NOT EXISTS decision_tree (
            id SERIAL PRIMARY KEY,
            user_queries TEXT[],
            response TEXT,
            embedding vector(384)  -- Use the dimension size of the SentenceTransformer model
        );
    """)
    conn.commit()
    conn.close()
