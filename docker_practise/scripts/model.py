from pydantic import BaseModel
from typing import List

# Pydantic models
class Query(BaseModel):
    text: str

class Node(BaseModel):
    user_queries: List[str]
    response: str
