from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import List
import os

class Settings(BaseSettings):
    """Application settings using Pydantic"""
    
    # Database Configuration
    database_url: str = "postgresql://paramaterial_user:paramaterial_password@localhost:5432/paramaterial"
    
    # Redis Configuration
    redis_url: str = "redis://localhost:6379/0"
    
    # Security Settings
    secret_key: str = "dev-secret-key-change-in-production"
    jwt_secret_key: str = "jwt-secret-key-change-in-production"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60
    refresh_token_expire_days: int = 30
    password_min_length: int = 8
    
    # File Upload Settings
    max_content_length: int = 52428800  # 50MB
    upload_folder: str = "uploads"
    allowed_extensions: List[str] = ["txt", "csv", "xlsx", "xls", "json"]
    
    # Application Settings
    items_per_page: int = 20
    mail_default_sender: str = "noreply@paramaterial.com"
    invitation_expiry_days: int = 7
    
    # Environment
    debug: bool = True
    testing: bool = False
    
    # CORS Settings
    cors_origins: List[str] = [
        "http://localhost:3000",
        "http://localhost:3001", 
        "https://paramaterial.vercel.app"
    ]
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
        # Map environment variables to settings
        fields = {
            'database_url': {'env': 'DATABASE_URL'},
            'redis_url': {'env': 'REDIS_URL'},
            'secret_key': {'env': 'SECRET_KEY'},
            'jwt_secret_key': {'env': 'JWT_SECRET_KEY'},
            'max_content_length': {'env': 'MAX_CONTENT_LENGTH'},
            'upload_folder': {'env': 'UPLOAD_FOLDER'},
            'items_per_page': {'env': 'ITEMS_PER_PAGE'},
            'mail_default_sender': {'env': 'MAIL_DEFAULT_SENDER'},
            'password_min_length': {'env': 'PASSWORD_MIN_LENGTH'},
            'invitation_expiry_days': {'env': 'INVITATION_EXPIRY_DAYS'},
            'debug': {'env': 'DEBUG'},
            'testing': {'env': 'TESTING'},
        }

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()