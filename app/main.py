import logging
import time
import uuid
import base64
from typing import Dict, Any, List

from fastapi import FastAPI, Request, HTTPException
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
import plotly.graph_objects as go
import plotly.io as pio
import uvicorn

# Import your project's modules
from core.agent import get_agent_executor
from core.db import initialize_db_pool, close_db_pool
from configs.settings import settings

# --- FastAPI App Setup ---
app = FastAPI(
    title="Aurobino Agent Server",
    description="A standalone API server for the stateful LangGraph agent.",
    version="1.0.0",
)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# This global variable will hold the compiled agent executor
AGENT_EXECUTOR = None

# --- Helper Functions ---
def convert_fig_to_base64(fig_json: str) -> str:
    """Converts a Plotly figure from its JSON representation to a base64 PNG image."""
    try:
        fig = pio.from_json(fig_json)
        img_bytes = pio.to_image(fig, format="png", scale=2)
        base64_string = base64.b64encode(img_bytes).decode('utf-8')
        return f"data:image/png;base64,{base64_string}"
    except Exception as e:
        logger.error(f"Failed to convert Plotly figure to image: {e}")
        return ""

# --- App Lifecycle Events ---
@app.on_event("startup")
async def startup_event():
    """
    Initializes all necessary resources when the API server starts.
    This includes the database pool and the compiled agent executor.
    """
    global AGENT_EXECUTOR
    logger.info("Server starting up...")
    await initialize_db_pool()

    # Build the correct connection string manually from its components
    correct_conn_string = (
        f"postgresql://{settings.DB_USER}:{settings.DB_PASSWORD}"
        f"@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"
    )
    
    checkpointer_context_manager = AsyncPostgresSaver.from_conn_string(correct_conn_string)
    app.state.checkpointer_cm = checkpointer_context_manager
    
    checkpointer_instance = await app.state.checkpointer_cm.__aenter__()

    # CORRECTION 1: Removed the incorrect .setup() call.
    # The database tables must be created manually one time using the provided SQL script.
    
    AGENT_EXECUTOR = get_agent_executor(checkpointer=checkpointer_instance)
    logger.info("Server startup complete. Agent executor is ready.")


@app.on_event("shutdown")
async def shutdown_event():
    """Closes resources gracefully when the API server shuts down."""
    logger.info("Server shutting down...")
    
    if hasattr(app.state, "checkpointer_cm"):
        await app.state.checkpointer_cm.__aexit__(None, None, None)

    await close_db_pool()
    logger.info("Server shutdown complete.")


# --- Primary API Endpoint ---
@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """
    This is the main endpoint that OpenWebUI will communicate with.
    It mimics the OpenAI Chat Completions API structure for seamless integration.
    """
    if not AGENT_EXECUTOR:
        raise HTTPException(status_code=503, detail="Agent is not available. Please check server logs.")

    body = await request.json()
    messages = body.get("messages", [])
    
    user_id = body.get("user", "default-user")
    thread_id = f"user-{user_id}"
    
    question = messages[-1].get("content") if messages else ""

    if not question:
        raise HTTPException(status_code=400, detail="No question found in the request.")

    config = {"configurable": {"thread_id": thread_id}}
    input_data = {"question": question, "user_id": user_id}
    final_state = {}

    try:
        async for event in AGENT_EXECUTOR.astream(input_data, config=config):
            for key, value in event.items():
                # CORRECTION 2: Check if the node output is not None before updating state.
                # This makes the loop more robust and prevents TypeErrors.
                if value:
                    final_state.update(value)
        
        summary = final_state.get("summary", "I was unable to process your request. Please try again.")
        figures_json = final_state.get("plotly_figure_json", [])
        
        for i, fig_json in enumerate(figures_json):
            base64_img = convert_fig_to_base64(fig_json)
            if base64_img:
                summary += f"\n\n![Chart {i+1}]({base64_img})"

        return {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "Aurobino Agent",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": summary,
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }
    except Exception as e:
        logger.error(f"An error occurred during agent execution for user '{user_id}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9099)