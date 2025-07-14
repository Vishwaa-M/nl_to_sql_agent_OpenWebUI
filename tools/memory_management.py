

import logging
from typing import Dict, Any

from langchain_core.tools import tool

from configs.settings import settings
from core.vector_store import get_vector_store_manager, MEMORY_COLLECTION_NAME

# Configure a dedicated logger for these tools
logger = logging.getLogger(__name__)

@tool
async def load_memory(user_id: str, query: str) -> str:
    """
    Retrieves the most relevant long-term memories for a given user by performing
    an efficient, filtered search in the vector database. This tool is used at the
    beginning of an interaction to provide the agent with personalized context.
    """
    logger.info(f"Tool 'load_memory' invoked for user_id: {user_id} with query: '{query[:100]}...'")
    if not user_id:
        logger.warning("No user_id provided to load_memory tool. Cannot retrieve memories.")
        return "No memories could be retrieved as the user is not identified."

    try:
        vector_store = get_vector_store_manager()
        
        # FIX 3: Use the top_k value from settings for easy tuning.
        top_k_value = settings.MEMORY_SEARCH_TOP_K

        # Construct a filter expression to search only for memories of the specific user.
        filter_expr = f'user_id == "{user_id}"'

        memory_results = await vector_store.similarity_search(
            collection_name=MEMORY_COLLECTION_NAME,
            query_text=query,
            top_k=top_k_value,
            filter_expression=filter_expr
        )
        
        if not memory_results:
            logger.info(f"No long-term memories found for user_id: {user_id}")
            return "No relevant long-term memories were found for this user."

        # Combine the retrieved memories into a single context string.
        formatted_memories = "\n- ".join([r['text'] for r in memory_results])
        logger.info(f"Successfully loaded {len(memory_results)} memories for user_id: {user_id}")
        
        return f"Here are some potentially relevant facts from past conversations:\n- {formatted_memories}"

    except Exception as e:
        logger.error(f"An unexpected error occurred in load_memory: {e}", exc_info=True)
        return f"Error: Failed to load long-term memory. Details: {e}"


@tool
async def save_memory(user_id: str, fact: str) -> str:
    """
    Saves a single, curated fact to the user's long-term memory, associating
    it with their user_id as metadata for future filtered retrieval.
    """
    logger.info(f"Tool 'save_memory' invoked for user_id: {user_id} with fact: '{fact}'")
    if not user_id or not fact:
        logger.warning("User ID or fact is missing. Cannot save memory.")
        return "Failed to save memory: User ID or fact was not provided."

    try:
        vector_store = get_vector_store_manager()
        
        # Create the metadata dictionary that will be stored alongside the memory text.
        memory_metadata = [{"user_id": user_id}]
        
        await vector_store.add_documents(
            collection_name=MEMORY_COLLECTION_NAME,
            texts=[fact],
            metadata=memory_metadata
        )
        
        logger.info(f"Successfully saved memory for user_id: {user_id}")
        return f"Successfully saved memory: '{fact}'"

    except Exception as e:
        logger.error(f"An unexpected error occurred in save_memory: {e}", exc_info=True)
        return f"Error: Failed to save memory. Details: {e}"

