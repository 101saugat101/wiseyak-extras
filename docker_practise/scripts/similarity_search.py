from fastapi import APIRouter, HTTPException
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel
from typing import List
from database import get_db_connection

# Initialize router and embedding model
router = APIRouter()
model = SentenceTransformer('all-MiniLM-L6-v2')  # Choose a suitable embedding model

# Pydantic models
class Query(BaseModel):
    text: str

class Node(BaseModel):
    user_queries: List[str]
    response: str

# API to add a node to the decision tree
@router.post("/add_node/")
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

# API to perform similarity search
@router.post("/similarity_search/")
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
