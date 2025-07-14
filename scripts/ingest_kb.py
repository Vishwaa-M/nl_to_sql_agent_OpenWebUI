import asyncio
import logging
import csv
from pathlib import Path
from typing import List, Dict

from core.db import get_db_connection, initialize_db_pool, close_db_pool
from core.vector_store import get_vector_store_manager, SCHEMA_COLLECTION_NAME, FEW_SHOT_COLLECTION_NAME
from configs.settings import settings

# Configure logging for the script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define the path to the knowledge base files
KNOWLEDGE_BASE_DIR = Path(__file__).parent.parent / "knowledge_base"
FEW_SHOT_CSV_PATH = KNOWLEDGE_BASE_DIR / "few_shot_examples.csv"

async def fetch_postgresql_schema() -> List[str]:
    """
    Connects to the PostgreSQL database and extracts a detailed, human-readable
    schema for all tables in the specified schema.
    """
    logger.info(f"Connecting to database to fetch schema from '{settings.DB_SCHEMA_NAME}'...")
    
    table_docs = []
    
    try:
        async with get_db_connection() as conn:
            tables_query = "SELECT table_name FROM information_schema.tables WHERE table_schema = %s;"
            tables = await conn.execute(tables_query, (settings.DB_SCHEMA_NAME,))
            table_names = [row['table_name'] for row in await tables.fetchall()]
            logger.info(f"Found {len(table_names)} tables: {table_names}")

            for table_name in table_names:
                table_def = f"Table: {table_name}\n"

                columns_query = """
                    SELECT column_name, data_type, is_nullable, column_default
                    FROM information_schema.columns
                    WHERE table_schema = %s AND table_name = %s
                    ORDER BY ordinal_position;
                """
                columns = await conn.execute(columns_query, (settings.DB_SCHEMA_NAME, table_name))
                table_def += "Columns:\n"
                for col in await columns.fetchall():
                    table_def += f"  - {col['column_name']} {col['data_type']} {'NULL' if col['is_nullable'] == 'YES' else 'NOT NULL'}{f' (default: {col['column_default']})' if col['column_default'] else ''}\n"

                pk_query = """
                    SELECT kcu.column_name
                    FROM information_schema.table_constraints tc
                    JOIN information_schema.key_column_usage kcu ON tc.constraint_name = kcu.constraint_name AND tc.table_schema = kcu.table_schema
                    WHERE tc.table_schema = %s AND tc.table_name = %s AND tc.constraint_type = 'PRIMARY KEY';
                """
                pk_result = await conn.execute(pk_query, (settings.DB_SCHEMA_NAME, table_name))
                pk_columns = [row['column_name'] for row in await pk_result.fetchall()]
                if pk_columns:
                    table_def += f"Primary Key: {', '.join(pk_columns)}\n"

                fk_query = """
                    SELECT kcu.column_name, ccu.table_name AS referenced_table, ccu.column_name AS referenced_column
                    FROM information_schema.table_constraints tc
                    JOIN information_schema.key_column_usage kcu ON tc.constraint_name = kcu.constraint_name AND tc.table_schema = kcu.table_schema
                    JOIN information_schema.constraint_column_usage ccu ON tc.constraint_name = ccu.constraint_name AND tc.table_schema = ccu.table_schema
                    WHERE tc.table_schema = %s AND tc.table_name = %s AND tc.constraint_type = 'FOREIGN KEY';
                """
                fk_result = await conn.execute(fk_query, (settings.DB_SCHEMA_NAME, table_name))
                fk_list = [f"{row['column_name']} references {row['referenced_table']}({row['referenced_column']})" for row in await fk_result.fetchall()]
                if fk_list:
                    table_def += "Foreign Keys:\n"
                    for fk in fk_list:
                        table_def += f"  - {fk}\n"

                index_query = "SELECT indexname, indexdef FROM pg_indexes WHERE schemaname = %s AND tablename = %s;"
                indexes = await conn.execute(index_query, (settings.DB_SCHEMA_NAME, table_name))
                index_list = [f"{idx['indexname']}: {idx['indexdef']}" for idx in await indexes.fetchall()]
                if index_list:
                    table_def += "Indexes:\n"
                    for idx in index_list:
                        table_def += f"  - {idx}\n"

                table_docs.append(table_def.strip())
        
        logger.info("Successfully fetched and formatted schema for all tables.")
        return table_docs
    except Exception as e:
        logger.error(f"Failed to fetch PostgreSQL schema: {e}", exc_info=True)
        return []

def load_few_shot_examples(file_path: Path) -> List[str]:
    """Loads question-query pairs from a CSV file."""
    if not file_path.exists():
        logger.warning(f"Few-shot examples file not found at: {file_path}")
        return []
    
    logger.info(f"Loading few-shot examples from {file_path}...")
    examples = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) == 2:
                    examples.append(f"Question: {row[0]}\nSQL Query: {row[1]}")
        logger.info(f"Successfully loaded {len(examples)} few-shot examples.")
        return examples
    except Exception as e:
        logger.error(f"Failed to load few-shot examples: {e}", exc_info=True)
        return []

async def main():
    """Main orchestration function to run the ingestion process."""
    logger.info("--- Starting Knowledge Base Ingestion Script ---")
    
    try:
        await initialize_db_pool()
        vector_store = get_vector_store_manager()

        logger.info("\n--- Processing Database Schema ---")
        schema_docs = await fetch_postgresql_schema()
        if schema_docs:
            # FIX: Correctly awaiting the async function.
            await vector_store.add_documents(
                collection_name=SCHEMA_COLLECTION_NAME,
                texts=schema_docs
            )
        else:
            logger.warning("No schema documents to ingest.")

        logger.info("\n--- Processing Few-Shot Examples ---")
        few_shot_docs = load_few_shot_examples(FEW_SHOT_CSV_PATH)
        if few_shot_docs:
            # FIX: Correctly awaiting the async function.
            await vector_store.add_documents(
                collection_name=FEW_SHOT_COLLECTION_NAME,
                texts=few_shot_docs
            )
        else:
            logger.warning("No few-shot examples to ingest.")

        logger.info("\n--- Knowledge Base Ingestion Complete ---")

    finally:
        await close_db_pool()
        logger.info("Database pool closed.")

if __name__ == "__main__":
    asyncio.run(main())
