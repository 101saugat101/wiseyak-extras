from fastapi import FastAPI
from similarity_search import router as similarity_router
from database import create_table
import uvicorn

# Initialize FastAPI app
app = FastAPI()

# Include the router for similarity search
app.include_router(similarity_router)

# Run table creation on startup
@app.on_event("startup")
def startup_event():
    create_table()

# Entry point for running the application
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
