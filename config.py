import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    db_url = os.getenv("DATABASE_URL")
    if db_url:
        db_url = db_url.replace("postgres://", "postgresql://")

    SQLALCHEMY_DATABASE_URI = db_url or "sqlite:///photo_share.db"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key")

    EMAIL_USER = os.getenv("EMAIL_USER")
    EMAIL_PASS = os.getenv("EMAIL_PASS")

    HF_TOKEN = os.getenv("HF_TOKEN")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")