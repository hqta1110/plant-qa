# Create a new file called config.py
import os
from pathlib import Path
from typing import List, Union, Optional
from pydantic import BaseSettings, Field
from dotenv import load_dotenv

# Load .env file if it exists
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env file"""
    
    # API Configuration
    API_HOST: str = Field("0.0.0.0", env="API_HOST")
    API_PORT: int = Field(8000, env="API_PORT")
    CORS_ORIGINS: Union[str, List[str]] = Field("*", env="CORS_ORIGINS")
    
    # Model Paths
    CLASSIFIER_PATH: str = Field("./models/dino_best.pth", env="CLASSIFIER_PATH")
    EMBED_PATH: str = Field("./models/embedding_best.pth", env="EMBED_PATH")
    LABELS_PATH: str = Field("./data/data", env="LABELS_PATH")
    METADATA_PATH: str = Field("./data/merge_metadata.json", env="METADATA_PATH")
    RETRIEVAL_INDEX_PATH: str = Field("./data/plant_index_2.pkl", env="RETRIEVAL_INDEX_PATH")
    
    # Hugging Face Integration
    HF_USE_HUB: bool = Field(False, env="HF_USE_HUB") 
    HF_MODEL_REPO: str = Field("", env="HF_MODEL_REPO")
    HF_DATASET_REPO: str = Field("", env="HF_DATASET_REPO")
    HF_TOKEN: Optional[str] = Field(None, env="HF_TOKEN")
    
    # GPU Configuration
    CUDA_VISIBLE_DEVICES: str = Field("0", env="CUDA_VISIBLE_DEVICES")
    
    # Application Settings
    DEBUG: bool = Field(False, env="DEBUG")
    LOG_LEVEL: str = Field("info", env="LOG_LEVEL")
    UPLOAD_DIR: str = Field("temp_uploads", env="UPLOAD_DIR")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

# Create settings instance
settings = Settings()

# For backward compatibility and ease of use, export settings to environment variables
os.environ["CUDA_VISIBLE_DEVICES"] = settings.CUDA_VISIBLE_DEVICES

# Create a model config dictionary for modules that expect it
MODEL_CONFIG = {
    "classifier_path": settings.CLASSIFIER_PATH,
    "embed_path": settings.EMBED_PATH,
    "labels_path": settings.LABELS_PATH,
    "metadata_path": settings.METADATA_PATH,
    "retrieval_index_file": settings.RETRIEVAL_INDEX_PATH
}