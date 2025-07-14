"""
This module defines the nodes of the LangGraph. Each node is a step in the
agent's reasoning process.

REFACTORING V10 (HITL Removal):
- The `verification_node` has been completely removed.
- The `query_execution_node` has been simplified to be fully autonomous. It no
  longer checks for user approval ('approve'/'reject') and directly executes
  the generated SQL query.
"""

import logging
import json
from typing import Dict, Any, List

from agents.states import AgentState
from core.llm import get_llm_client
from prompts.system_prompts import (
    ROUTER_PROMPT_TEMPLATE,
    SQL_GENERATION_PROMPT_TEMPLATE,
    SQL_CORRECTION_PROMPT_TEMPLATE,
    MEMORY_CURATION_PROMPT_TEMPLATE,
)

from tools.sql_tools import get_relevant_schema, get_few_shot_examples, execute_sql_query
from tools.analysis_tools import summarize_results, plan_visualizations, create_plotly_figure
from tools.memory_management import load_memory, save_memory

# Configure a dedicated logger for this module
logger = logging.getLogger(__name__)

# --- Core Node Implementations ---

async def router_node(state: AgentState) -> Dict[str, Any]:
    """
    Classifies the user's question to decide the execution path.
    """
    logger.info("--- Node: Routing ---")
    question = state.get("question")
    llm_client = get_llm_client()
    chat_history_str = "\n".join([f"{msg.type}: {msg.content}" for msg in state.get("chat_history", [])])
    prompt = ROUTER_PROMPT_TEMPLATE.format(question=question, chat_history=chat_history_str)
    messages = [{"role": "user", "content": prompt}]
    response_str = await llm_client.invoke_chat_completion(messages, json_mode=True)
    routing_result = json.loads(response_str)
    next_node = routing_result.get("route", "general_conversation")
    logger.info(f"Router decision: '{next_node}'")
    return {"next_node": next_node}

async def direct_response_node(state: AgentState) -> Dict[str, Any]:
    """
    Handles general conversation by providing a direct LLM response.
    """
    logger.info("--- Node: Direct Response ---")
    llm_client = get_llm_client()
    chat_history_str = "\n".join([f"{msg.type}: {msg.content}" for msg in state.get("chat_history", [])])
    prompt = (f"You are a helpful AI assistant. Respond to the user's latest message conversationally. "
              f"Context:\n{chat_history_str}\n\nUser's message: '{state['question']}'")
    messages = [{"role": "system", "content": prompt}]
    response = await llm_client.invoke_chat_completion(messages)
    return {"summary": response}

async def schema_linking_node(state: AgentState) -> Dict[str, Any]:
    """
    Gathers all necessary context (schema, few-shot examples) for SQL generation.
    """
    logger.info("--- Node: Schema Linking ---")
    question = state.get("question")
    schema = await get_relevant_schema.ainvoke({"question": question})
    examples = await get_few_shot_examples.ainvoke({"question": question})
    return {"retrieved_schema": schema, "few_shot_examples": examples}

async def query_generation_node(state: AgentState) -> Dict[str, Any]:
    """
    Generates the SQL query using the gathered context.
    """
    logger.info("--- Node: SQL Query Generation ---")
    llm_client = get_llm_client()
    prompt = SQL_GENERATION_PROMPT_TEMPLATE.format(
        question=state["question"],
        schema=state["retrieved_schema"],
        few_shot_examples=state["few_shot_examples"],
        long_term_memory=state.get("long_term_memory", "No long-term memory available for this user.")
    )
    messages = [{"role": "user", "content": prompt}]
    sql_query = await llm_client.invoke_chat_completion(messages)
    cleaned_sql = sql_query.strip().replace("```sql", "").replace("```", "").strip()
    logger.info(f"Successfully generated SQL: {cleaned_sql}")
    return {"generated_sql": cleaned_sql}

# The `verification_node` has been removed entirely.

async def query_execution_node(state: AgentState) -> Dict[str, Any]:
    """
    Executes the generated SQL query directly without human verification.
    """
    logger.info("--- Node: Executing SQL Query ---")
    # The logic for checking user_response has been removed.
    sql_query = state.get("generated_sql")
    result = await execute_sql_query.ainvoke({"sql_query": sql_query})

    # The node now simply returns the result of the query execution.
    # The state fields for HITL (`awaiting_verification`, `user_response`) are no longer managed here.
    return {
        "query_result": result.get("query_result"),
        "sql_error": result.get("sql_error"),
        "generated_sql": sql_query,
    }

async def self_correction_node(state: AgentState) -> Dict[str, Any]:
    """
    Constructs a prompt for the LLM to correct its own SQL query based on a DB error.
    """
    logger.info("--- Node: Attempting SQL Self-Correction ---")
    llm_client = get_llm_client()
    prompt = SQL_CORRECTION_PROMPT_TEMPLATE.format(
        question=state["question"],
        failed_sql=state["generated_sql"],
        error_message=state["sql_error"],
        schema=state["retrieved_schema"]
    )
    messages = [{"role": "user", "content": prompt}]
    corrected_sql = await llm_client.invoke_chat_completion(messages)
    cleaned_sql = corrected_sql.strip().replace("```sql", "").replace("```", "").strip()
    logger.info(f"Successfully generated corrected SQL: {cleaned_sql}")
    return {"generated_sql": cleaned_sql, "sql_error": None}

async def summarization_node(state: AgentState) -> Dict[str, Any]:
    """
    Generates a natural language summary of the query results.
    """
    logger.info("--- Node: Summarizing Results ---")
    query_result = state.get("query_result")
    if not query_result:
        return {"summary": "The query returned no data to summarize."}
    summary = await summarize_results.ainvoke({"question": state["question"], "query_result": query_result})
    return {"summary": summary}

# --- Visualization Nodes (REFACTORED) ---

async def visualization_planning_node(state: AgentState) -> Dict[str, Any]:
    """
    Calls the tool to create a JSON plan for the dashboard visualizations.
    """
    logger.info("--- Node: Planning Visualizations ---")
    query_result = state.get("query_result")
    if not query_result:
        logger.warning("No data available to plan a visualization.")
        return {"visualization_plan": None}
    plan_result = await plan_visualizations.ainvoke({"question": state["question"], "data": query_result})
    return {"visualization_plan": plan_result.get("plan")}

async def figure_generation_node(state: AgentState) -> Dict[str, Any]:
    """
    Programmatically generates a list of Plotly Figure objects and
    serializes them to JSON for state persistence.
    """
    logger.info("--- Node: Generating Plotly Figures ---")
    plan = state.get("visualization_plan")
    data = state.get("query_result")
    if not plan or not plan.get("charts") or not data:
        logger.warning("No visualization plan or data found. Skipping figure generation.")
        return {"plotly_figure_json": []}
    
    figures_as_json = []
    for chart_plan in plan["charts"]:
        result = await create_plotly_figure.ainvoke({"chart_plan": chart_plan, "data": data})
        if figure := result.get("figure"):
            figures_as_json.append(figure.to_json())
        else:
            logger.error(f"Failed to generate figure for chart '{chart_plan.get('title')}': {result.get('error')}")
            
    logger.info(f"Successfully generated and serialized {len(figures_as_json)} Plotly figure(s).")
    return {"plotly_figure_json": figures_as_json}

# --- Memory Management Nodes (Unchanged) ---

async def load_memory_node(state: AgentState) -> Dict[str, Any]:
    """
    Loads long-term memory for the user at the start of the SQL flow.
    """
    logger.info("--- Node: Loading Long-Term Memory ---")
    user_id = state.get("user_id")
    question = state.get("question")
    if not user_id:
        logger.warning("No user_id in state. Skipping memory load.")
        return {"long_term_memory": "Long-term memory is disabled for this session."}
    memory_context = await load_memory.ainvoke({"user_id": user_id, "query": question})
    return {"long_term_memory": memory_context}

async def curate_memory_node(state: AgentState) -> Dict[str, Any]:
    """
    Analyzes the conversation to decide which facts are worth saving.
    """
    logger.info("--- Node: Curating Memory ---")
    chat_history = state.get("chat_history")
    if not chat_history:
        return {"facts_to_save": []}
    try:
        llm_client = get_llm_client()
        chat_history_str = "\n".join([f"{msg.type}: {msg.content}" for msg in chat_history])
        prompt = MEMORY_CURATION_PROMPT_TEMPLATE.format(chat_history=chat_history_str)
        messages = [{"role": "user", "content": prompt}]
        response_str = await llm_client.invoke_chat_completion(messages, json_mode=True)
        if not response_str:
            raise ValueError("Memory curation LLM call returned no response.")
        curation_result = json.loads(response_str)
        facts = curation_result.get("facts_to_save", [])
        logger.info(f"Curated {len(facts)} facts to save to long-term memory.")
        return {"facts_to_save": facts}
    except Exception as e:
        logger.error(f"Error during memory curation: {e}", exc_info=True)
        return {"facts_to_save": []}

async def save_memory_node(state: AgentState) -> Dict[str, Any]:
    """
    Saves the curated facts to the vector store, one by one.
    """
    logger.info("--- Node: Saving Facts to Memory ---")
    user_id = state.get("user_id")
    facts_to_save = state.get("facts_to_save")
    if not user_id or not facts_to_save:
        logger.warning("No user_id or facts to save. Skipping memory persistence.")
        return {}
    for fact in facts_to_save:
        await save_memory.ainvoke({"user_id": user_id, "fact": fact})
    return {}