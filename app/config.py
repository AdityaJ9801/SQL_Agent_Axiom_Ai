from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional, Literal


class Settings(BaseSettings):
    LLM_PROVIDER: Literal["ollama", "openai", "anthropic", "groq", "grok", "azure_openai"] = "openai"
    GROQ_API_KEY: Optional[str] = ""
    XAI_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4o"
    GROQ_MODEL: str = "llama-3.1-8b-instant"
    OLLAMA_MODEL: str = ""
    OLLAMA_BASE_URL: str = ""
    
    # Azure OpenAI
    AZURE_OPENAI_API_KEY: Optional[str] = None
    AZURE_OPENAI_ENDPOINT: str = ""
    AZURE_OPENAI_DEPLOYMENT_NAME: str = ""
    AZURE_OPENAI_API_VERSION: str = "2024-02-15-preview"
    
    DB_DIALECT: str = "duckdb"
    DATABASE_URL: Optional[str] = None
    DUCKDB_DATA_PATH: str = "./data"
    
    BIGQUERY_PROJECT_ID: Optional[str] = None
    BIGQUERY_DATASET: Optional[str] = None
    
    # Azure Blob Storage — used by DuckDBConnector to authenticate az:// URLs
    AZURE_STORAGE_CONNECTION_STRING: Optional[str] = None

    USE_DOCKER: bool = False
    USE_EC2: bool = False
    PORT: int = 8002
    
    MAX_RESULT_ROWS: int = 1000

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()
