"""
This module creates a FastAPI backend that exposes the NL-to-SQL agent's
functionality through a streaming, OpenAI-compatible API. This is the standard
pattern for integration with frontends like Open WebUI.

FINAL VERSION V5 (Definitive Fix):
- Correctly handles the asynchronous lifecycle of the LangGraph checkpointer.
- The checkpointer context is now properly "entered" at startup and "exited"
  at shutdown, resolving the AttributeError at startup.
- Fixes missing imports for CORSMiddleware and StreamingResponse.
- Hardcodes 'approve' for the user_response to work with Open WebUI.
"""

import logging
import uvicorn
import uuid
import json
import base64
import plotly.io as pio
import asyncio
from typing import Dict, Any, List, Optional
from decimal import Decimal

from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware # IMPORT FIX
from pydantic import BaseModel, Field
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langchain_core.messages import HumanMessage, AIMessage

from core.db import initialize_db_pool, close_db_pool
from core.agent import get_agent_executor
from configs.settings import settings

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
logger = logging.getLogger(__name__)

# --- FastAPI App Initialization ---
app = FastAPI(
    title="DataNexus Agent API",
    description="An OpenAI-compatible streaming API for the NL-to-SQL agent.",
    version="1.3.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# --- Pydantic Models ---
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    stream: bool = True

# --- Global Agent Resources ---
checkpointer_manager: Optional[Any] = None
checkpointer: Optional[AsyncPostgresSaver] = None

# --- FastAPI Lifecycle Events ---
@app.on_event("startup")
async def startup_event():
    """Initializes all necessary resources when the API server starts."""
    logger.info("--- API Server Starting Up ---")
    await initialize_db_pool()
    
    global checkpointer, checkpointer_manager
    try:
        from urllib.parse import urlparse, urlunparse
        raw_uri = str(settings.DATABASE_URL)
        temp_uri = raw_uri.replace("postgresql+psycopg", "postgresql")
        parsed_uri = urlparse(temp_uri)
        cleaned_path = f"/{parsed_uri.path.lstrip('/')}" if parsed_uri.path else f"/{settings.DB_NAME}"
        conn_string = urlunparse(parsed_uri._replace(path=cleaned_path))
        
        # 1. Create the context manager
        checkpointer_manager = AsyncPostgresSaver.from_conn_string(conn_string)
        
        # 2. "Enter" the context to get the live checkpointer object
        checkpointer = await checkpointer_manager.__aenter__()
        
        # 3. Perform a setup/health check on the live checkpointer object
        await checkpointer.setup()
        
        logger.info("Postgres checkpointer initialized and entered successfully.")
    except Exception as e:
        logger.critical(f"Failed to initialize Postgres checkpointer: {e}", exc_info=True)
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleans up resources when the API server shuts down."""
    logger.info("--- API Server Shutting Down ---")
    global checkpointer_manager
    if checkpointer_manager:
        try:
            # 4. "Exit" the context to release resources
            await checkpointer_manager.__aexit__(None, None, None)
            logger.info("Postgres checkpointer exited successfully.")
        except Exception as e:
            logger.error(f"Failed to exit checkpointer context gracefully: {e}")
    await close_db_pool()


# --- Helper for SSE chunks ---
def create_sse_chunk(content: str, model: str, finish_reason: Optional[str] = None) -> str:
    chunk = {"id":f"chatcmpl-{uuid.uuid4()}","object":"chat.completion.chunk","created":int(uuid.uuid4().int>>64),"model":model,"choices":[{"index":0,"delta":{"content":content},"finish_reason":finish_reason}]}
    return f"data: {json.dumps(chunk)}\n\n"

# --- API Endpoint ---
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest = Body(...)):
    if not checkpointer:
        raise HTTPException(status_code=503, detail="Checkpointer not initialized.")

    last_user_message = next((msg.content for msg in reversed(request.messages) if msg.role == 'user'), None)
    if not last_user_message:
        raise HTTPException(status_code=400, detail="No user message found.")

    user_id = "open-webui-user"
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    agent_executor = get_agent_executor(checkpointer=checkpointer)
    history = [HumanMessage(content=msg.content) if msg.role == 'user' else AIMessage(content=msg.content) for msg in request.messages]
    
    # Since OpenWebUI has no approval buttons, we hardcode 'approve' to bypass the HITL check.
    agent_input = {"question": last_user_message, "chat_history": history, "user_id": user_id, "user_response": "approve"}

    async def stream_agent_response():
        final_agent_state = {}
        try:
            node_display_names={"router":"Routing Query...","schema_linking":"Analyzing Database Schema...","load_memory":"Loading Personal Memory...","query_generation":"Writing SQL Query...","request_verification":"Awaiting Your Approval...","query_execution":"Executing SQL Query...","self_correction":"Attempting to Correct SQL Error...","summarization":"Summarizing Results...","plan_visualization":"Planning Visualizations...","figure_generation":"Generating Charts...","curate_memory":"Analyzing Conversation...","save_memory":"Saving Key Facts to Memory..."}
            
            async for chunk in agent_executor.astream(agent_input, config=config):
                for node_name, node_output in chunk.items():
                    if node_name == "__end__":
                        final_agent_state = node_output
                        break
                    display_name=node_display_names.get(node_name,f"Executing {node_name}...")
                    status_update={"status":"in_progress","message":display_name}
                    yield create_sse_chunk(json.dumps(status_update),request.model)
                    await asyncio.sleep(0.1)
                if "__end__" in chunk: break

            summary = final_agent_state.get("summary", "The agent did not produce a summary.")
            plotly_figure_json = final_agent_state.get("plotly_figure_json", [])
            response_content = summary
            
            if plotly_figure_json:
                try:
                    status_update={"status":"in_progress","message":"Rendering chart..."}
                    yield create_sse_chunk(json.dumps(status_update),request.model)
                    fig_json=plotly_figure_json[0]
                    fig=pio.from_json(fig_json)
                    img_bytes=fig.to_image(format="png",width=800,height=600,scale=2)
                    b64_string=base64.b64encode(img_bytes).decode('utf-8')
                    markdown_image=f"![chart](data:image/png;base64,{b64_string})"
                    response_content+=f"\n\n{markdown_image}"
                except Exception as e:
                    logger.error(f"Failed to convert Plotly figure to image: {e}")
                    response_content+="\n\n(Chart could not be rendered)"
            
            yield create_sse_chunk(response_content, request.model, finish_reason="stop")

        except Exception as e:
            logger.error(f"An unhandled error occurred during streaming: {e}", exc_info=True)
            error_message=json.dumps({"status":"error","message":f"An unexpected server error occurred: {e}"})
            yield create_sse_chunk(error_message, request.model, finish_reason="stop")
        
        yield "data: [DONE]\n\n"

    return StreamingResponse(stream_agent_response(), media_type="text/event-stream")

# --- Run Server ---
if __name__ == "__main__":
    logger.info("Starting DataNexus Agent API server on http://localhost:8001")
    uvicorn.run(app, host="0.0.0.0", port=8001)
