import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from urllib.parse import urlparse, urlunparse

from psycopg import AsyncConnection
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool, PoolTimeout

from configs.settings import settings

# Configure a dedicated logger for this module for better observability
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Global Pool Variable ---
# This variable will hold the single, application-wide connection pool.
# It is initialized to None and populated by the `initialize_db_pool` function.
db_pool: AsyncConnectionPool = None


async def initialize_db_pool(startup_retries: int = 5, retry_delay: int = 3):
    """
    Initializes the asynchronous database connection pool with retry logic.

    This is a critical function that should be called during the application's
    startup sequence (e.g., in a FastAPI startup event). It creates the
    connection pool and performs an initial health check.

    Args:
        startup_retries (int): Number of times to retry connecting on startup.
        retry_delay (int): Seconds to wait between retries.
    """
    global db_pool
    logger.info("Initializing database connection pool...")

    # FIX: Robustly parse and clean the connection URL.
    try:
        # First, ensure the scheme is 'postgresql'
        temp_uri = str(settings.DATABASE_URL).replace("postgresql+psycopg", "postgresql")
        
        # Parse the URL into its components
        parsed_uri = urlparse(temp_uri)
        
        # Clean the path component (the database name) by removing any leading slashes
        cleaned_path = f"/{parsed_uri.path.lstrip('/')}"
        
        # Rebuild the URI with the cleaned path
        conninfo = urlunparse(parsed_uri._replace(path=cleaned_path))
        
        # Log a confirmation without printing credentials.
        logger.info("Successfully parsed and sanitized the database connection string.")

    except Exception as e:
        logger.critical(f"Failed to parse DATABASE_URL: {e}", exc_info=True)
        raise ValueError("Invalid DATABASE_URL format") from e


    for attempt in range(1, startup_retries + 1):
        # The pool object is created but not yet opened
        pool = None
        try:
            pool = AsyncConnectionPool(
                conninfo=conninfo,
                min_size=settings.DB_POOL_MIN,
                max_size=settings.DB_POOL_MAX,
                timeout=settings.DB_QUERY_TIMEOUT,
                kwargs={"row_factory": dict_row},
                # Defer opening to an explicit call below
                open=False
            )

            # Explicitly open the pool to avoid RuntimeWarning.
            await pool.open()
            
            # Perform a quick check to ensure the pool is operational
            await pool.check()

            db_pool = pool  # Assign the live pool to the global variable
            logger.info(
                "Database connection pool initialized successfully. "
                f"min_size={settings.DB_POOL_MIN}, max_size={settings.DB_POOL_MAX}"
            )
            return  # The pool remains open

        except Exception as e:
            logger.error(
                f"Attempt {attempt}/{startup_retries} to initialize database pool failed: {e}"
            )
            # If a pool object was created, ensure it's closed on failure
            if pool and not pool.closed:
                await pool.close()

            if attempt < startup_retries:
                logger.info(f"Retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
            else:
                logger.critical("Could not initialize database pool after multiple retries. Exiting.")
                raise ConnectionError("Failed to initialize database pool.") from e


@asynccontextmanager
async def get_db_connection() -> AsyncGenerator[AsyncConnection, None]:
    """
    Provides a database connection from the pool as an async context manager.
    """
    if db_pool is None:
        logger.critical("Database pool is not initialized. Call 'initialize_db_pool' first.")
        raise ConnectionError("Database pool is not initialized.")
    
    try:
        async with db_pool.connection() as conn:
            logger.debug("Acquired database connection from pool.")
            yield conn
    except PoolTimeout:
        logger.error(f"Could not get a connection from the pool within {settings.DB_QUERY_TIMEOUT}s timeout.")
        raise
    finally:
        logger.debug("Released database connection back to pool.")


async def close_db_pool():
    """
    Closes the database connection pool gracefully.
    """
    global db_pool
    if db_pool:
        logger.info("Closing database connection pool...")
        await db_pool.close()
        db_pool = None
        logger.info("Database connection pool closed.")


async def health_check() -> bool:
    """
    Performs a health check on the database.
    """
    if db_pool is None:
        logger.warning("Health check failed: Database pool is not initialized.")
        return False
        
    try:
        await db_pool.check()
        logger.info("Database health check successful.")
        return True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return False


if __name__ == "__main__":
    async def main():
        # This main block is for direct testing of this module
        await initialize_db_pool()
        
        is_healthy = await health_check()
        print(f"Database Health Status: {'OK' if is_healthy else 'FAIL'}")

        if is_healthy:
            try:
                # Example: Fetching the current time from the database
                async with get_db_connection() as conn:
                    async with conn.cursor() as cur:
                        await cur.execute("SELECT NOW() as current_time;")
                        result = await cur.fetchone()
                        print(f"Successfully fetched data from DB: {result}")
            except Exception as e:
                print(f"An error occurred during DB operation: {e}")

        await close_db_pool()

    asyncio.run(main())
