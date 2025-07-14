"""
This module defines the complete, enterprise-grade LangGraph agent.

REFACTORING V10 (HITL Removal):
- The Human-in-the-Loop (HITL) verification flow has been removed to create a
  fully autonomous agent compatible with non-interactive environments like an API.
- The `verification_node` and its associated conditional logic have been deleted.
- The graph now follows a direct path from `query_generation` to `query_execution`.
"""

import logging
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.pregel import Pregel

from agents.states import AgentState
from agents.nodes import (
    router_node,
    direct_response_node,
    schema_linking_node,
    query_generation_node,
    # verification_node has been removed
    query_execution_node,
    self_correction_node,
    summarization_node,
    visualization_planning_node,
    figure_generation_node,
    load_memory_node,
    curate_memory_node,
    save_memory_node,
)
from configs.settings import settings

logger = logging.getLogger(__name__)

# --- Conditional Edge Functions ---

def should_continue_to_summarization(state: AgentState) -> str:
    """Conditional edge: Route to self-correction on error, else summarize."""
    return "self_correction" if state.get("sql_error") else "summarization"

def should_continue_to_visualization(state: AgentState) -> str:
    """Conditional edge: Proceed to visualization if there's data, else go to memory curation."""
    return "plan_visualization" if state.get("query_result") else "curate_memory"

# The `after_sql_generation` conditional edge function has been removed.

def create_agent_graph() -> StateGraph:
    """
    Builds the autonomous agent graph definition.
    """
    logger.info("--- Defining Autonomous Agent Graph ---")

    workflow = StateGraph(AgentState)

    logger.info("Registering all nodes with the graph...")
    workflow.add_node("router", router_node)
    workflow.add_node("direct_response", direct_response_node)
    workflow.add_node("load_memory", load_memory_node)
    workflow.add_node("schema_linking", schema_linking_node)
    workflow.add_node("query_generation", query_generation_node)
    # The `request_verification` node is no longer added to the graph.
    workflow.add_node("query_execution", query_execution_node)
    workflow.add_node("self_correction", self_correction_node)
    workflow.add_node("summarization", summarization_node)
    workflow.add_node("plan_visualization", visualization_planning_node)
    workflow.add_node("figure_generation", figure_generation_node)
    workflow.add_node("curate_memory", curate_memory_node)
    workflow.add_node("save_memory", save_memory_node)

    logger.info("Wiring the graph with edges and conditional logic...")
    workflow.set_entry_point("router")

    workflow.add_conditional_edges(
        "router",
        lambda s: s["next_node"],
        {"general_conversation": "direct_response", "sql_query": "load_memory"}
    )
    workflow.add_edge("direct_response", "curate_memory")

    # SQL Path (Now fully autonomous)
    workflow.add_edge("load_memory", "schema_linking")
    workflow.add_edge("schema_linking", "query_generation")

    # The conditional edge for verification has been replaced with a direct edge.
    workflow.add_edge("query_generation", "query_execution")

    workflow.add_conditional_edges(
        "query_execution",
        should_continue_to_summarization,
        {"self_correction": "self_correction", "summarization": "summarization"}
    )
    
    # The self-correction loop remains unchanged.
    workflow.add_edge("self_correction", "query_execution")

    # Visualization Path
    workflow.add_conditional_edges(
        "summarization",
        should_continue_to_visualization,
        {"plan_visualization": "plan_visualization", "curate_memory": "curate_memory"}
    )
    workflow.add_edge("plan_visualization", "figure_generation")
    workflow.add_edge("figure_generation", "curate_memory")

    # Final Memory Path
    workflow.add_edge("curate_memory", "save_memory")
    workflow.add_edge("save_memory", END)

    logger.info("Agent graph definition complete.")
    return workflow

def get_agent_executor(checkpointer: AsyncPostgresSaver) -> Pregel:
    """
    Compiles the agent graph with the provided checkpointer into a runnable executor.
    """
    logger.info("Compiling agent graph with the provided checkpointer...")
    try:
        graph = create_agent_graph()
        app = graph.compile(checkpointer=checkpointer)
        logger.info("Autonomous agent executor compiled successfully.")
        return app
    except Exception as e:
        logger.error(f"Failed to compile agent graph: {e}", exc_info=True)
        raise