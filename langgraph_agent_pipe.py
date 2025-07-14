import os
import asyncio
import base64
import logging
from typing import Dict, Any, Union

import plotly.graph_objects as go
import plotly.io as pio
from pydantic import BaseModel
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

# Import your existing agent and db functions
from core.agent import get_agent_executor
from core.db import initialize_db_pool
from configs.settings import settings

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Pipeline:
    """
    This class defines the OpenWebUI Pipe for the fully autonomous, stateful
    LangGraph Data Analyst Agent.
    """
    class Valves(BaseModel):
        # Valves can be used to pass pipeline-specific settings from the UI
        # or environment variables. For now, we will use settings from the config file.
        pass

    def __init__(self):
        """
        Initializes the agent's dependencies, including the database pool and
        the agent executor with its Postgres checkpointer. This runs only once
        when the pipeline server starts.
        """
        self.agent_executor = None
        try:
            # The agent's components rely on an active asyncio event loop.
            # We ensure the DB pool is initialized within that context.
            asyncio.run(initialize_db_pool())
            
            # The checkpointer is crucial for the agent's stateful memory
            conn_string = settings.DATABASE_URL
            self.checkpointer = AsyncPostgresSaver.from_conn_string(conn_string)
            self.agent_executor = get_agent_executor(checkpointer=self.checkpointer)
            
            logger.info("LangGraph Agent Pipe initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize LangGraph Agent Pipe: {e}", exc_info=True)

    def _convert_fig_to_base64(self, fig_json: str) -> str:
        """Converts a Plotly figure from JSON to a base64 PNG image."""
        try:
            fig = go.from_json(fig_json)
            # Use kaleido to convert the figure to a static image with good resolution
            img_bytes = pio.to_image(fig, format="png", scale=2)
            base64_string = base64.b64encode(img_bytes).decode('utf-8')
            return f"data:image/png;base64,{base64_string}"
        except Exception as e:
            logger.error(f"Failed to convert Plotly figure: {e}")
            return ""

    async def pipe(self, body: Dict, __user: Dict) -> str:
        """
        This is the main method called by OpenWebUI for each user message.
        It runs the autonomous agent and returns the final response as a string.
        """
        if not self.agent_executor:
            return "Error: Agent executor is not initialized. Please check the pipeline logs."

        # Extract relevant info from the request body
        messages = body.get("messages", [])
        question = messages[-1].get("content", "") if messages else ""
        
        # The conversation_id from OpenWebUI is used as the thread_id for LangGraph's
        # stateful checkpointer. This is how memory is maintained across requests.
        thread_id = body.get("conversation_id", "default-thread")

        config = {"configurable": {"thread_id": thread_id}}
        input_data = {"question": question, "user_id": thread_id}
        final_state = {}

        try:
            # Asynchronously stream the agent's execution from start to finish
            async for event in self.agent_executor.astream(input_data, config=config):
                for key, value in event.items():
                    final_state.update(value)

            # The agent now runs to completion without pausing.
            # We format the final output after the stream ends.
            summary = final_state.get("summary", "No summary was generated.")
            figures_json = final_state.get("plotly_figure_json", [])
            
            # Convert any generated figures to images and embed them in the summary
            for i, fig_json in enumerate(figures_json):
                base64_img = self._convert_fig_to_base64(fig_json)
                if base64_img:
                    summary += f"\n\n![Chart {i+1}]({base64_img})"
            
            return summary

        except Exception as e:
            logger.error(f"An error occurred during agent execution: {e}", exc_info=True)
            return f"An error occurred while processing your request: {e}"