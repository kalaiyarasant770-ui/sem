
# config.py
import os
from dotenv import load_dotenv
from pathlib import Path
from urllib.parse import quote_plus
import re
import logging


# Load .env from project root (go up one level from src/)
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

# Verify it loaded
if not os.getenv("CHILLER_DB_HOST"):
    print("❌ ERROR: .env file not found or empty!")
    print(f"   Looking for: {env_path}")
    print(f"   Exists: {env_path.exists()}")
else:
    print(f"✅ .env loaded: DB_HOST={os.getenv('CHILLER_DB_HOST')[:30]}...")

# Load environment variables
load_dotenv()

# LLM Configuration
OPENAI_MODEL = "gpt-3.5-turbo" 
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Database Configuration
DB_USER = os.getenv("CHILLER_DB_USER", "chatbot_readonly")
DB_PASSWORD = os.getenv("CHILLER_DB_PWD")
DB_HOST = os.getenv("CHILLER_DB_HOST")
DB_PORT = os.getenv("CHILLER_DB_PORT")
DB_NAME = os.getenv("CHILLER_DB_NAME")

# Encode password if needed
if re.search(r'[^A-Za-z0-9]', DB_PASSWORD):
    ENCODED_PASSWORD = quote_plus(DB_PASSWORD)
else:
    ENCODED_PASSWORD = DB_PASSWORD

DATABASE_URL = f"mysql+pymysql://{DB_USER}:{ENCODED_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Redis Configuration
REDIS_URL = os.getenv("UPSTASH_REDIS_REST_URL")
REDIS_TOKEN = os.getenv("UPSTASH_REDIS_REST_TOKEN")

# Memory Settings
MAX_HISTORY_MESSAGES = 30
SESSION_EXPIRY = 86400  # 24 hours

# RAG Settings
VECTOR_DB_PATH = "rag_data"
CHUNK_SIZE = 50
CHUNK_OVERLAP = 5
EMBEDDINGS_MODEL = "BAAI/bge-small-en-v1.5"

# SQL Safety
FORBIDDEN_KEYWORDS = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'TRUNCATE', 'CREATE', 'REPLACE', 'MERGE', 'SET', 'LOAD']

# Business Constants
TARIFF_RATE = 10.0  #10.0

COLUMN_MAPPING={}

# In config.py — replace or append to existing SCHEMA_CONTEXT
SCHEMA_CONTEXT = """
DATABASE SCHEMA:

Table: vertical
  - vertical_Id (INT, PK)
  - vertical_name (VARCHAR)
  - isdelete (BOOLEAN, default 0)

Table: cluster
  - cluster_id (INT, PK)
  - cluster_name (VARCHAR)
  - vertical_Id (INT, FK → vertical.vertical_id)
  - isdelete (BOOLEAN, default 0)

Table: devices
  - device_Id (INT, PK)
  - device_name (VARCHAR)
  - cluster_id (INT, FK → cluster.cluster_id)
  - isdelete (BOOLEAN, default 0)

Table: device_data (Today's data, per minute)
  - device_Id (INT, FK)
  - datetime (DATETIME)
  - device_value (JSON) → V18: power, V19: temp_in, V20: temp_out

Table: historical_data_minute (Current month)
  - device_id (INT, FK)
  - updatedtime (DATETIME)
  - raw_value (JSON)

Table: historical_data_minute_YYYY_MM (Past months)
  - Same as above

RULES:
- Always use device_id in queries
- Use vertical_id → cluster → devices to resolve device_id
- Match device_name in user query (case-insensitive)
- JSON access: JSON_UNQUOTE(device_value->'$.V18')
"""

# System Prompt for LLM
SYSTEM_PROMPT = """
You are a SQL query generator for READ-ONLY database access to chiller monitoring systems.

INTENT CLASSIFICATION - BE VERY PRECISE:

1. 'metrics' - ONLY for queries requesting SPECIFIC DATA or NUMBERS
2. 'advisory' - ONLY for queries requesting ADVICE or GUIDANCE
3. 'greeting' - ONLY for casual/personal conversation

SQL GENERATION RULES (only for 'metrics' intent):
- ONLY generate SELECT queries
- Always use parameterized queries (:device_id, :start, :end)
- Valid device_ids: 1195 (Chiller 1), 1201 (Chiller 2), 1202 (Chiller 3)
"""

logging.info("✅ Configuration loaded successfully")