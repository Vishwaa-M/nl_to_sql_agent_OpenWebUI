"""
This module provides a singleton VectorStoreManager for interacting with Milvus,
a vector database used to store and retrieve database schema information,
few-shot examples, and long-term user memory for SQL query generation.
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional

from pymilvus import (
    connections,
    utility,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType
)
from sentence_transformers import SentenceTransformer

from configs.settings import settings

# Configure a dedicated logger for this module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Global Vector Store Manager ---
vector_store_manager: Optional['VectorStoreManager'] = None

# --- Constants ---
SCHEMA_COLLECTION_NAME = "db_schema_metadata"
FEW_SHOT_COLLECTION_NAME = "few_shot_sql_examples"
MEMORY_COLLECTION_NAME = "long_term_user_memory"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


class VectorStoreManager:
    """
    An enterprise-grade manager for interacting with a Milvus vector database.
    
    This class handles metadata, enabling efficient, filtered queries
    at the database level, which is critical for production environments.
    """

    def __init__(self):
        self._connect()
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        self._initialize_collections()

    def _connect(self):
        """Establishes a connection to the Milvus server."""
        try:
            logger.info(f"Connecting to Milvus at {settings.MILVUS_HOST}:{settings.MILVUS_PORT}...")
            connections.connect(
                alias="default",
                host=settings.MILVUS_HOST,
                port=str(settings.MILVUS_PORT),
                user=settings.MILVUS_USER or None,
                db_name=settings.MILVUS_DB_NAME
            )
            logger.info(f"Successfully connected to Milvus database '{settings.MILVUS_DB_NAME}'.")
        except Exception as e:
            logger.critical(f"Failed to connect to Milvus: {e}", exc_info=True)
            raise

    def _initialize_collections(self):
        """Ensures that all necessary collections, including memory, exist."""
        self._create_standard_collection(SCHEMA_COLLECTION_NAME)
        self._create_standard_collection(FEW_SHOT_COLLECTION_NAME)
        self._create_memory_collection(MEMORY_COLLECTION_NAME)

    def _create_standard_collection(self, collection_name: str):
        """Creates a standard collection without user-specific metadata."""
        if utility.has_collection(collection_name):
            Collection(collection_name).load()
            logger.info(f"Standard collection '{collection_name}' already exists and loaded.")
            return

        logger.info(f"Standard collection '{collection_name}' not found. Creating...")
        doc_id = FieldSchema(name="doc_id", dtype=DataType.INT64, is_primary=True, auto_id=True)
        text_content = FieldSchema(name="text_content", dtype=DataType.VARCHAR, max_length=4096)
        embedding = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)
        schema = CollectionSchema(fields=[doc_id, text_content, embedding], description=f"Collection for {collection_name}")
        collection = Collection(name=collection_name, schema=schema)
        collection.create_index(field_name="embedding", index_params={"metric_type": "L2", "index_type": "IVF_FLAT", "params": {"nlist": 1024}})
        collection.load()
        logger.info(f"Standard collection '{collection_name}' created and loaded.")

    def _create_memory_collection(self, collection_name: str):
        """Creates the memory collection with a dedicated user_id field for filtering."""
        if utility.has_collection(collection_name):
            Collection(collection_name).load()
            logger.info(f"Memory collection '{collection_name}' already exists and loaded.")
            return
            
        logger.info(f"Memory collection '{collection_name}' not found. Creating with user_id field...")
        doc_id = FieldSchema(name="doc_id", dtype=DataType.INT64, is_primary=True, auto_id=True)
        user_id = FieldSchema(name="user_id", dtype=DataType.VARCHAR, max_length=255)
        text_content = FieldSchema(name="text_content", dtype=DataType.VARCHAR, max_length=4096)
        embedding = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)
        schema = CollectionSchema(fields=[doc_id, user_id, text_content, embedding], description=f"Collection for {collection_name}")
        collection = Collection(name=collection_name, schema=schema)
        collection.create_index(field_name="embedding", index_params={"metric_type": "L2", "index_type": "IVF_FLAT", "params": {"nlist": 1024}})
        collection.create_index(field_name="user_id")
        collection.load()
        logger.info(f"Memory collection '{collection_name}' created, indexed, and loaded.")

    # FIX: Made the method asynchronous to be properly awaited by the tools.
    async def add_documents(self, collection_name: str, texts: List[str], metadata: Optional[List[Dict[str, Any]]] = None):
        """
        Embeds documents and inserts them into a collection asynchronously.
        """
        if not texts:
            logger.warning("No documents provided to add.")
            return

        logger.info(f"Embedding and adding {len(texts)} documents to '{collection_name}'...")
        
        # This function contains blocking IO/CPU-bound code, so we run it in an executor.
        def _blocking_add_documents():
            embeddings = self.embedding_model.encode(texts, show_progress_bar=False)
            
            entities = []
            if metadata and len(metadata) == len(texts):
                for text, emb, meta in zip(texts, embeddings, metadata):
                    entities.append({"text_content": text, "embedding": emb, **meta})
            else:
                for text, emb in zip(texts, embeddings):
                    entities.append({"text_content": text, "embedding": emb})

            collection = Collection(collection_name)
            collection.insert(entities)
            collection.flush()
        
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, _blocking_add_documents)
            logger.info(f"Successfully added {len(texts)} documents to '{collection_name}'.")
        except Exception as e:
            logger.error(f"Failed to add documents to '{collection_name}': {e}", exc_info=True)

    # FIX: Made the method asynchronous to be properly awaited by the tools.
    async def similarity_search(self, collection_name: str, query_text: str, top_k: int = 5, filter_expression: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Performs an asynchronous similarity search with an optional metadata filter.
        """
        logger.info(f"Performing similarity search in '{collection_name}' with filter: '{filter_expression}'")

        # This function contains blocking IO/CPU-bound code, so we run it in an executor.
        def _blocking_search():
            query_embedding = self.embedding_model.encode([query_text])[0]
            collection = Collection(collection_name)
            
            results = collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param={"metric_type": "L2", "params": {"nprobe": 10}},
                limit=top_k,
                expr=filter_expression,
                output_fields=["text_content"]
            )
            
            return [{"text": hit.entity.get('text_content'), "score": hit.distance} for hit in results[0]]

        try:
            loop = asyncio.get_running_loop()
            formatted_results = await loop.run_in_executor(None, _blocking_search)
            logger.info(f"Found {len(formatted_results)} results from filtered search.")
            return formatted_results
        except Exception as e:
            logger.error(f"Similarity search failed in '{collection_name}': {e}", exc_info=True)
            return []


def get_vector_store_manager() -> 'VectorStoreManager':
    """Initializes and returns the singleton VectorStoreManager instance."""
    global vector_store_manager
    if vector_store_manager is None:
        logger.info("Creating a new singleton instance of the VectorStoreManager.")
        vector_store_manager = VectorStoreManager()
    return vector_store_manager
