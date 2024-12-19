import psycopg2
from psycopg2.extras import RealDictCursor
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

# Initialize FastAPI and embedding model
app = FastAPI()
model = SentenceTransformer('all-MiniLM-L6-v2')  # Choose a suitable embedding model

# Database connection details
DB_CONFIG = {
    "dbname": "similarity_search",
    "user": "postgres",
    "password": "heheboii420",
    "host": "localhost",
    "port": 5432
}

# Pydantic models
class Query(BaseModel):
    text: str

class Node(BaseModel):
    user_queries: List[str]
    response: str

# Database utility functions
def get_db_connection():
    try:
        conn = psycopg2.connect(**DB_CONFIG, cursor_factory=RealDictCursor)
        return conn
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database connection error: {e}")

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

# API to add a node to the decision tree
@app.post("/add_node/")
def add_node(node: Node):
    conn = get_db_connection()
    cursor = conn.cursor()

    # Generate embedding for the representative query
    representative_query = node.user_queries[0]  # Use the first query as the representative
    embedding = model.encode(representative_query).tolist()

    # Insert data into the database
    cursor.execute("""
        INSERT INTO decision_tree (user_queries, response, embedding)
        VALUES (%s, %s, %s);
    """, (node.user_queries, node.response, embedding))
    conn.commit()
    conn.close()
    return {"message": "Node added successfully!"}

@app.post("/similarity_search/")
def similarity_search(query: Query):
    conn = get_db_connection()
    cursor = conn.cursor()

    # Generate embedding for the input query
    input_embedding = model.encode(query.text).tolist()

    # Convert the embedding to a format that PostgreSQL understands as a vector
    input_embedding_str = f"[{', '.join(map(str, input_embedding))}]"

    # Perform similarity search in PostgreSQL using pgvector similarity operator
    cursor.execute(f"""
        SELECT response, (embedding <=> '{input_embedding_str}'::vector) AS similarity
        FROM decision_tree
        ORDER BY similarity
        LIMIT 1;
    """)
    result = cursor.fetchone()
    conn.close()

    if result:
        return {"response": result["response"], "similarity": result["similarity"]}
    else:
        return {"message": "No matching response found."}

# Run table creation on startup
@app.on_event("startup")
def startup_event():
    create_table()
