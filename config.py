from __future__ import annotations

import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from openai import OpenAI
from sqlalchemy import create_engine, text


load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./app.db")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_data")
PROCESSING_MODEL = os.getenv("PROCESSING_MODEL", "gpt-4o-mini")
LINKING_MODEL = os.getenv("LINKING_MODEL", "gpt-4o-mini")
RETRIEVAL_MODEL = os.getenv("RETRIEVAL_MODEL", "text-embedding-3-small")
INTERACTION_MODEL = os.getenv("INTERACTION_MODEL", "gpt-4o-mini")
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "gpt-4o-mini")
USER_ID_HEADER = os.getenv("USER_ID_HEADER", "X-User-ID")
ENABLE_COST_LOGGING = os.getenv("ENABLE_COST_LOGGING", "true").lower() == "true"
CACHE_PROCESSED_SOURCES = os.getenv("CACHE_PROCESSED_SOURCES", "true").lower() == "true"
DEPLOY_ENV = os.getenv("DEPLOY_ENV", "production")

if not OPENAI_API_KEY:
    raise RuntimeError(
        "Missing OPENAI_API_KEY environment variable. "
        "Set it in .env before starting the server."
    )

openai_client = OpenAI(api_key=OPENAI_API_KEY)
db_engine = create_engine(DATABASE_URL, future=True)

Path(CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)
chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)


def run_startup_checks() -> None:
    with db_engine.connect() as connection:
        connection.execute(text("SELECT 1"))

    chroma_client.list_collections()