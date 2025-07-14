import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, PostgresDsn, ValidationError
from typing import Optional


class Settings(BaseSettings):
    """
    Main settings class to hold all configuration for the application.
    It validates and loads settings from environment variables.
    """
    # --- Database Configuration ---
    DB_HOST: str = Field(..., description="Database host address")
    DB_PORT: int = Field(5432, description="Database port number")
    DB_NAME: str = Field(..., description="Database name")
    DB_USER: str = Field(..., description="Database username")
    DB_PASSWORD: str = Field(..., description="Database password")
    DB_SSL_MODE: str = Field("prefer", description="Database SSL mode")
    DB_SCHEMA_NAME: str = Field("public", description="Default database schema")
     
    # --- Database Pool Configuration ---
    DB_POOL_MIN: int = Field(2, description="Minimum connections in the pool")
    DB_POOL_MAX: int = Field(20, description="Maximum connections in the pool")
    DB_QUERY_TIMEOUT: int = Field(10, description="Database query timeout in seconds")

    # --- Mistral AI Configuration ---
    MISTRAL_API_KEY: str = Field(..., description="API key for Mistral AI services")
    MISTRAL_MODEL_NAME: str = Field("mistral-large-latest", description="The specific Mistral model to use")

    # --- Milvus Vector Store Configuration ---
    MILVUS_HOST: str = Field("localhost", description="Milvus vector store host address")
    MILVUS_PORT: int = Field(19530, description="Milvus vector store port number")
    MILVUS_DB_NAME: str = Field("default", description="The database name to use in Milvus")
    MILVUS_USER: Optional[str] = Field(None, description="Optional username for Milvus")
    MILVUS_PASSWORD: Optional[str] = Field(None, description="Optional password for Milvus")
    
    # --- Agent Behavior Configuration ---
    SCHEMA_SEARCH_TOP_K: int = Field(5, description="Number of relevant schema snippets to retrieve")
    FEW_SHOT_TOP_K: int = Field(3, description="Number of few-shot examples to retrieve")
    MEMORY_SEARCH_TOP_K: int = Field(5, description="Number of relevant memories to retrieve for a user")

    DATABASE_URL: Optional[str] = None
    
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=False,
        extra='ignore' 
    )
    
    def __init__(self, **values):
        """Custom init to construct DATABASE_URL after validation."""
        super().__init__(**values)
        self.DATABASE_URL = self._assemble_db_connection()

    def _assemble_db_connection(self) -> str:
        """Constructs the database connection string from individual components."""
        return str(PostgresDsn.build(
            scheme="postgresql+psycopg",
            username=self.DB_USER,
            password=self.DB_PASSWORD,
            host=self.DB_HOST,
            port=self.DB_PORT,
            # FIX: Removed the leading slash. Pydantic's builder handles it.
            path=self.DB_NAME
        ))

# Create a singleton instance of the settings.
try:
    settings = Settings()
except ValidationError as e:
    print("ERROR: Could not load application settings. Please check your .env file and environment variables.")
    print(e)
    raise SystemExit("Configuration validation failed.") from e

if __name__ == "__main__":
    print("--- Application Settings Loaded Successfully ---")
    print(f"Database Host: {settings.DB_HOST}")
    print(f"Database Name: {settings.DB_NAME}")
    print(f"Database URL: {settings.DATABASE_URL}")
    print(f"Mistral Model: {settings.MISTRAL_MODEL_NAME}")
    if settings.MISTRAL_API_KEY:
        print(f"Mistral API Key Loaded: {'*' * 12}{settings.MISTRAL_API_KEY[-4:]}")
    print("\n--- Agent Behavior Settings ---")
    print(f"Schema Search K-Value: {settings.SCHEMA_SEARCH_TOP_K}")
    print(f"Few-Shot Search K-Value: {settings.FEW_SHOT_TOP_K}")
    print(f"Memory Search K-Value: {settings.MEMORY_SEARCH_TOP_K}")
    print("------------------------------------------")
