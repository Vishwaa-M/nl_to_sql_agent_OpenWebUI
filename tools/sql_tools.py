
import logging
from typing import Dict, Any, List
from langchain_core.tools import tool
from psycopg import Error as PsycopgError
from core.db import get_db_connection
from core.vector_store import get_vector_store_manager, SCHEMA_COLLECTION_NAME, FEW_SHOT_COLLECTION_NAME

# Configure a dedicated logger for these tools
logger = logging.getLogger(__name__)

@tool
async def get_relevant_schema(question: str) -> str:
    """
    Analyzes a user's natural language question and retrieves the most relevant
    database schema definitions (table names, columns, types) from the vector store.
    """
    logger.info(f"Tool 'get_relevant_schema' invoked with question: '{question[:100]}...'")
    try:
        vector_store = get_vector_store_manager()
        schema_results = await vector_store.similarity_search(
            collection_name=SCHEMA_COLLECTION_NAME,
            query_text=question,
            top_k=5
        )
        if not schema_results:
            logger.warning("No relevant schema found for the question.")
            return "No relevant database schema information was found for your question."
        retrieved_schema = "\n\n---\n\n".join([r['text'] for r in schema_results])
        logger.info("Successfully retrieved relevant schema.")
        return retrieved_schema
    except Exception as e:
        logger.error(f"An unexpected error occurred in get_relevant_schema: {e}", exc_info=True)
        return f"Error: Failed to retrieve database schema. Details: {e}"

@tool
async def get_few_shot_examples(question: str) -> str:
    """
    Analyzes a user's natural language question and retrieves the most similar
    question-SQL query pairs from the vector store.
    """
    logger.info(f"Tool 'get_few_shot_examples' invoked with question: '{question[:100]}...'")
    try:
        vector_store = get_vector_store_manager()
        example_results = await vector_store.similarity_search(
            collection_name=FEW_SHOT_COLLECTION_NAME,
            query_text=question,
            top_k=3
        )
        if not example_results:
            logger.warning("No few-shot examples found for the question.")
            return "No query examples were found for this type of question."
        few_shot_examples = "\n\n---\n\n".join([r['text'] for r in example_results])
        logger.info("Successfully retrieved few-shot examples.")
        return few_shot_examples
    except Exception as e:
        logger.error(f"An unexpected error occurred in get_few_shot_examples: {e}", exc_info=True)
        return f"Error: Failed to retrieve few-shot examples. Details: {e}"

@tool
async def execute_sql_query(sql_query: str) -> Dict[str, Any]:
    """
    Executes a read-only SQL query against the database and returns the results.
    This tool is for data retrieval only. It strictly expects a `SELECT` statement.
    """
    logger.info(f"Tool 'execute_sql_query' invoked with query: '{sql_query}'")

    if not isinstance(sql_query, str) or not sql_query.strip():
        return {"query_result": None, "sql_error": "Invalid or empty SQL query provided."}
    
    # Simple security check remains. The database is the true validator.
    if not sql_query.strip().upper().startswith("SELECT"):
        error_message = "Security Error: Only SELECT queries are allowed."
        logger.error(error_message)
        return {"query_result": None, "sql_error": error_message}


    try:
        async with get_db_connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(sql_query)

                if cur.description is None:
                    logger.info("Query executed successfully but returned no data/columns.")
                    return {"query_result": [], "sql_error": None}
                
                dict_results = await cur.fetchall()
                
                logger.info(f"Query executed successfully. Fetched {len(dict_results)} rows.")
                return {"query_result": dict_results, "sql_error": None}
                
    except PsycopgError as e:
        error_message = f"Database Execution Error: {e}"
        logger.error(error_message, exc_info=True)
        return {"query_result": None, "sql_error": str(e)}
    
    except Exception as e:
        error_message = f"An unexpected error occurred during query execution: {e}"
        logger.error(error_message, exc_info=True)
        return {"query_result": None, "sql_error": str(e)}
