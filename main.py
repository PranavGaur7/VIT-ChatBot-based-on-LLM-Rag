# main.py
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
import logging

# Import the necessary functions from your chat.py file
from chat import (
    initialize_llm,
    initialize_vectorstore,
    process_csvs_to_vectorstore,
    generate_answer,
    logger  # You can also import your configured logger
)

# --- Configuration ---
# Define the paths to your CSV files
# This matches the default in your argparse
DEFAULT_CSV_PATHS = [
    "result.csv",
    "faculty_data.csv",
    "VIT_Alumni_Data_10000.csv",
    "VIT_Originated_Startups_100.csv",
    "VIT_Research_Publications_Indian_2000.csv"
]
# Set to True to force-rebuild the DB on every server start
# (Good for development, bad for production)
FORCE_REBUILD_ON_START = False

# --- Global Variables ---
# These will hold the initialized models and be shared across requests
app_state = {
    "llm": None,
    "vector_store": None
}

# --- FastAPI Lifespan (Startup/Shutdown) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the application's lifespan.
    Initializes models on startup and cleans up on shutdown.
    """
    logger.info("Starting server and initializing models...")
    
    # Initialize LLM and Vector Store
    app_state["llm"] = initialize_llm()
    app_state["vector_store"] = initialize_vectorstore()
    
    # Process the CSVs into the vector store
    # This runs once when the server starts
    logger.info(f"Processing CSVs: {DEFAULT_CSV_PATHS}")
    process_csvs_to_vectorstore(
        csv_paths=DEFAULT_CSV_PATHS,
        vector_store=app_state["vector_store"],
        force_rebuild=FORCE_REBUILD_ON_START
    )
    
    logger.info("Startup complete. Models are loaded and ready.")
    
    yield  # The application runs here
    
    # --- Shutdown ---
    logger.info("Shutting down...")
    app_state["llm"] = None
    app_state["vector_store"] = None
    logger.info("Shutdown complete.")

# --- Initialize FastAPI App ---
app = FastAPI(
    title="Academic Assistant API",
    description="API for the VIT Academic Assistant RAG bot",
    lifespan=lifespan
)

# --- CORS Middleware ---
# This allows your frontend (running on a different port or file://)
# to communicate with this backend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (change in production)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# --- Pydantic Models for Request/Response ---
class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    answer: str

# --- API Endpoints ---
@app.get("/")
def read_root():
    """Root endpoint to check if the API is running."""
    return {"status": "Academic Assistant API is running"}

@app.post("/api/chat", response_model=ChatResponse)
async def post_chat(request: ChatRequest):
    """
    The main chat endpoint.
    Receives a query and returns the RAG-generated answer.
    """
    llm = app_state.get("llm")
    vector_store = app_state.get("vector_store")

    if not llm or not vector_store:
        logger.error("Models are not initialized.")
        raise HTTPException(status_code=503, detail="Service unavailable: Models are not initialized.")
    
    try:
        logger.info(f"Received query: {request.query}")
        
        # Use the generate_answer function from your chat.py
        answer = generate_answer(
            query=request.query,
            llm=llm,
            vector_store=vector_store
        )
        
        logger.info("Successfully generated answer.")
        return ChatResponse(answer=answer)
    
    except Exception as e:
        logger.error(f"Error during answer generation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

# --- Run the server ---
if __name__ == "__main__":
    # This allows you to run the server with: python main.py
    uvicorn.run(app, host="127.0.0.1", port=8000)