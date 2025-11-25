
# memory.py
import json
import uuid
import logging
import time
from upstash_redis import Redis
from config import REDIS_URL, REDIS_TOKEN, MAX_HISTORY_MESSAGES, SESSION_EXPIRY

# Initialize Redis client with detailed logging
t_redis_start = time.time()
logging.info("🔄 Initializing Redis connection...")

try:
    if REDIS_URL and REDIS_TOKEN:
        redis_client = Redis(url=REDIS_URL, token=REDIS_TOKEN)
        redis_client.ping()
        logging.info(f"✅ Redis connected successfully in {time.time() - t_redis_start:.2f} seconds")
    else:
        redis_client = None
        logging.warning("⚠️  Redis credentials not found - running without memory")
except Exception as e:
    redis_client = None
    logging.warning(f"⚠️  Redis connection failed: {e} - running without memory")

logging.info(f"Memory module initialized in {time.time() - t_redis_start:.2f} seconds")

def generate_session_id() -> str:
    """Generate a unique session ID."""
    session_id = str(uuid.uuid4())
    logging.info(f"🆕 Generated new session ID: {session_id[:8]}...")
    return session_id

def get_conversation_history(session_id: str) -> list:
    """Load conversation history from Redis."""
    if not redis_client or not session_id:
        logging.debug("No Redis client or session_id, returning empty history")
        return []
    try:
        history_json = redis_client.get(f"session:{session_id}")
        if history_json:
            history = json.loads(history_json)
            logging.info(f"📜 Loaded {len(history)} messages for session {session_id[:8]}...")
            return history
        logging.debug(f"No history found for session {session_id[:8]}...")
        return []
    except Exception as e:
        logging.error(f"❌ Failed to load history for session {session_id[:8]}...: {e}")
        return []

def save_conversation_history(session_id: str, history: list) -> bool:
    """Save conversation history to Redis."""
    if not redis_client or not session_id:
        logging.debug("No Redis client or session_id, skipping save")
        return False
    try:
        if len(history) > MAX_HISTORY_MESSAGES:
            removed_count = len(history) - MAX_HISTORY_MESSAGES
            history = history[-MAX_HISTORY_MESSAGES:]
            logging.info(f"🗑️  Trimmed {removed_count} old messages (keeping last {MAX_HISTORY_MESSAGES})")
        
        redis_client.setex(
            f"session:{session_id}",
            SESSION_EXPIRY,
            json.dumps(history)
        )
        logging.info(f"💾 Saved {len(history)} messages for session {session_id[:8]}... (expires in {SESSION_EXPIRY}s)")
        return True
    except Exception as e:
        logging.error(f"❌ Failed to save history for session {session_id[:8]}...: {e}")
        return False

def add_message_to_history(session_id: str, role: str, content: str) -> list:
    """Add a message to conversation history."""
    logging.debug(f"📝 Adding {role} message to session {session_id[:8]}...")
    history = get_conversation_history(session_id)
    history.append({"role": role, "content": content})
    
    if len(history) > MAX_HISTORY_MESSAGES:
        removed = history.pop(0)
        logging.info(f"🗑️  Removed oldest message (FIFO): {removed['role']} - {removed['content'][:50]}...")
    
    save_conversation_history(session_id, history)
    logging.debug(f"✅ Message added. Total messages: {len(history)}")
    return history

def clear_conversation_history(session_id: str) -> bool:
    """Clear conversation history for a session."""
    if not redis_client or not session_id:
        logging.warning("Cannot clear: No Redis client or session_id")
        return False
    try:
        redis_client.delete(f"session:{session_id}")
        logging.info(f"🗑️  Cleared all history for session {session_id[:8]}...")
        return True
    except Exception as e:
        logging.error(f"❌ Failed to clear history for session {session_id[:8]}...: {e}")
        return False