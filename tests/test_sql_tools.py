"""
test_sql_tools.py

This module tests the execute_sql_query tool from sql_tools.py by running a specific
SQL query against the database and logging the results. It uses the actual database
connection without mocking.
"""

import asyncio
import logging
from typing import Dict, Any

from core.db import initialize_db_pool, close_db_pool
from tools.sql_tools import execute_sql_query

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_execute_sql_query():
    """
    Tests the execute_sql_query tool with a specific query and logs the results.
    """
    logger.info("Starting test for execute_sql_query tool")

    # Define the test query
    test_query = "SELECT safety_stock FROM alert_product WHERE vendor_name = 'Jasmin Palace Trading LLC';"

    try:
        # Initialize the database pool
        logger.info("Initializing database connection pool for test")
        await initialize_db_pool()

        # Execute the query
        logger.info(f"Executing test query: {test_query}")
        result = await execute_sql_query.ainvoke({"sql_query": test_query})

        # Log the results
        if result["sql_error"]:
            logger.error(f"Query execution failed: {result['sql_error']}")
        else:
            query_result = result["query_result"]
            if query_result:
                logger.info(f"Query executed successfully. Retrieved {len(query_result)} rows:")
                for row in query_result:
                    logger.info(f"Row: {row}")
            else:
                logger.info("Query executed successfully but returned no rows.")

    except Exception as e:
        logger.error(f"Test failed due to unexpected error: {e}", exc_info=True)

    finally:
        # Clean up the database pool
        logger.info("Closing database connection pool")
        await close_db_pool()

if __name__ == "__main__":
    # Run the test
    asyncio.run(test_execute_sql_query())