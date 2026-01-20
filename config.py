import sys
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Windows Asyncio Fix
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Configuration
# MODEL_NAME = "gpt-oss-120b"  # or "Meta-Llama-3-70B-Instruct"
MODEL_NAME = "Qwen3-32B"  # or "Meta-Llama-3-70B-Instruct"
DB_PATH = "chat_sessions.db"

# Shared Memory (Mutable Dictionary)
GLOBAL_MEMORY = {
    "current_csv": None,
    "schema_info": "Unknown. Load CSV first.",
    "current_session_id": None,
    "excel_info": None
}