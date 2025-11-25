

#25-11-2025 an - kalaiyarasan770@gmail.com claude





#24-11-2025 = aftrenoon


# main.py
import logging
import time
import base64
from pathlib import Path
from datetime import datetime
from flask import Flask, request, jsonify, Response
from flask_cors import CORS

# === CORE IMPORTS ===
import json
from openai import OpenAI
from config import OPENAI_API_KEY, OPENAI_MODEL, TARIFF_RATE, COLUMN_MAPPING, SYSTEM_PROMPT, SCHEMA_CONTEXT
from sqlalchemy import text
from database import engine

# === DYNAMIC DEVICE SYSTEM ===
from typing import Dict
import pandas as pd
from threading import Lock

# === MODULES ===
from utils import generate_excel_report_in_memory, parse_recommendation_response
from chatbot import chatbot_graph, ChatbotState, classify_query_intent
import chatbot as chatbot_module
import re
from auth import authenticate_user
import logging

from parameter_query import detect_parameter_query, check_exit_command, detect_time_only,detect_list_parameters_query

# Configure logging FIRST
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[logging.StreamHandler()]
)
# Force all loggers to INFO level
logging.getLogger().setLevel(logging.INFO)
# === GLOBAL CACHE ===
VERTICAL_DEVICE_CACHE: Dict[int, pd.DataFrame] = {}
PARAMETER_CACHE: Dict[int, dict] = {}   # {vertical_id: {"df_params": df, "pin_context": str}}
CACHE_LOCK = Lock()
PARAM_CACHE: Dict[int, pd.DataFrame] = {}

# === PARAMETER / PIN CACHE ===
VERTICAL_PARAM_CACHE: Dict[int, pd.DataFrame] = {}

# ============================================================
# LOAD PARAMETERS FOR VERTICAL  (FINAL WORKING VERSION)
# ============================================================
def load_parameters_for_vertical(vertical_id: int, df_devices: pd.DataFrame) -> pd.DataFrame:
    """
    Build parameter DF:
        device_id | device_name | pin | parameter
    Extracted only from datastream.address JSON.
    Works for address-1, address-2, ..., dynamic.
    """

    logging.info(f"🔍 Loading datastream parameters for vertical_id={vertical_id}")

    try:
        # 1. Get clusters under this vertical
        with engine.connect() as conn:
            cluster_rows = conn.execute(
                text("SELECT cluster_id FROM cluster WHERE vertical_Id=:vid AND isdelete=0"),
                {"vid": vertical_id}
            ).fetchall()

        cluster_ids = [row[0] for row in cluster_rows]

        if not cluster_ids:
            logging.warning(f"⚠ No clusters found for vertical {vertical_id}")
            return pd.DataFrame(columns=["device_id", "device_name", "pin", "parameter"])

        # 2. Load datastream rows for all clusters
        with engine.connect() as conn:
            ds_rows = conn.execute(
                text("""
                    SELECT cluster_Id, address
                    FROM datastream
                    WHERE cluster_Id IN :cl_ids AND isdelete = 0
                """),
                {"cl_ids": tuple(cluster_ids)}
            ).fetchall()

        # Device lookup → {1223: "450kW Furnace", ...}
        device_lookup = df_devices.set_index("device_id")["device_name"].to_dict()
        all_rows = []
        # 3. Process each datastream JSON
        for cluster_id, address_json in ds_rows:
            if not address_json:
                continue
            try:
                address_blocks = json.loads(address_json)
            except:
                continue
            if not isinstance(address_blocks, list):
                continue
            for block in address_blocks:
                if not isinstance(block, dict):
                    continue
                pin = block.get("pin")
                if not pin:
                    continue
                # ✅ FIX: Handle both cases - address-X as keys AND address-X inside block
                address_sections = {}

                # Case 1: address-X as direct keys in block
                for key, section in block.items():
                    if key.startswith("address-") and isinstance(section, dict):
                        address_sections[key] = section

                # Case 2: Fallback - if no address-X keys found, check if block itself is address data
                if not address_sections and block.get("params"):
                    address_sections["address-1"] = block

                # Now process all found address sections
                for addr_key, section in address_sections.items():
                    if not isinstance(section, dict):
                        continue
                    parameter_name = section.get("params")
                    devices_list = section.get("devices")

                    if not parameter_name or not isinstance(devices_list, list):
                        continue

                    # map per device
                    for dev_id in devices_list:
                        if dev_id in device_lookup:
                            all_rows.append({
                                "device_id": dev_id,
                                "device_name": device_lookup[dev_id],
                                "pin": pin,
                                "parameter": parameter_name
                            })
        # Build DF
        df_params = pd.DataFrame(all_rows)
        # Deduplicate
        if not df_params.empty:
            df_params = df_params.drop_duplicates()

        logging.info(f"📋 PARAMETERS DF for vertical {vertical_id}:")
        logging.info("\n" + str(df_params.head(50)))

        logging.info(f"✅ Loaded {len(df_params)} parameter mappings for vertical {vertical_id}")

        return df_params

    except Exception as e:
        logging.error(f"❌ Failed to load parameters for vertical {vertical_id}: {e}")
        return pd.DataFrame(columns=["device_id", "device_name", "pin", "parameter"])



# === DEVICE TYPE MAPPING (INT → NAME) ===
DEVICE_TYPE_MAP = {
    0: "Outcomer",
    1: "Chiller",
    2: "Water",
    3: "Temperature",
    4: "Main Incomer",
    5: "DG",
    6: "Others"
}

def load_devices_for_vertical(vertical_id: int) -> pd.DataFrame:
    with CACHE_LOCK:
        if vertical_id in VERTICAL_DEVICE_CACHE:
            logging.info(f"CACHE HIT: vertical_id={vertical_id}")
            return VERTICAL_DEVICE_CACHE[vertical_id]

    # Only log when loading fresh (not from cache)
    logging.info(f"\n{'='*80}")
    logging.info(f"🔍 LOADING DEVICES FOR VERTICAL_ID={vertical_id}")
    logging.info(f"{'='*80}")
    
    sql = """
    SELECT 
        c.cluster_id, c.cluster_name, 
        d.device_id, d.device_name, d.device_type, d.slave_id
    FROM devices d
    JOIN cluster c ON d.cluster_id = c.cluster_id
    WHERE c.vertical_id = :vertical_id AND c.isdelete = 0 AND d.isdelete = 0
    ORDER BY c.cluster_name, d.device_name
    """
    try:
        with engine.connect() as conn:
            df = pd.read_sql(text(sql), conn, params={"vertical_id": vertical_id})

        if df.empty:
            logging.warning(f"NO DEVICES FOUND for vertical_id={vertical_id}")
            return df
        else:
            df['device_type_name'] = df['device_type'].map(DEVICE_TYPE_MAP).fillna("Unknown")
            df = df[['cluster_name', 'cluster_id', 'device_name', 'device_type_name', 'device_type', 'device_id', 'slave_id']]
            # Print full device table
            logging.info(f"✅ LOADED {len(df)} DEVICES:")
            logging.info("\n" + df[['cluster_name', 'cluster_id', 'device_name', 'device_type_name', 'device_type', 'device_id']].to_string(index=False))
            logging.info(f"{'='*80}\n")

        with CACHE_LOCK:
            VERTICAL_DEVICE_CACHE[vertical_id] = df
        return df

    except Exception as e:
        logging.error(f"FAILED to load devices: {e}")
        return pd.DataFrame()

def resolve_device_from_df_with_llm(df: pd.DataFrame, query: str, conversation_history: list = None) -> tuple:
    """
    Universal two-stage LLM device resolution.
    """
    if df.empty:
        return None, None, None, None, None

    logging.info(f"\n{'='*80}")
    logging.info(f"🔍 TWO-STAGE DEVICE RESOLUTION")
    logging.info(f"{'='*80}")
    logging.info(f"User Query: '{query}'")
    
    # ✅ BUILD DEVICE LIST FIRST (NEW)
    device_list = []
    for _, row in df.iterrows():
        device_list.append({
            'device_id': int(row['device_id']),
            'device_name': row['device_name'],
            'device_type': row['device_type_name'],
            'cluster': row['cluster_name']
        })
    
    devices_summary = "\n".join([f"- {d['device_name']} ({d['device_type']}, Cluster: {d['cluster']})" for d in device_list])
    
    logging.info(f"\n✅ AVAILABLE DEVICES IN VERTICAL ({len(device_list)} total):")
    logging.info(devices_summary)
    
    # Build conversation context
    context = ""
    if conversation_history:
        context = "\n\nRecent Conversation:\n"
        for msg in conversation_history[-6:]:
            role = msg.get('role', 'user')
            content = msg.get('content', '')[:150]
            context += f"{role}: {content}...\n"
    
    # ==================== STAGE 1: UNDERSTANDING ====================
    logging.info(f"\n--- STAGE 1: Understanding User Intent ---")

    stage1_prompt = f"""
You are a smart query analyzer. Extract EXACTLY what device the user is asking about.

CRITICAL: These are the ONLY devices available in this vertical:
{devices_summary}

{context}

Current Query: "{query}"

EXTRACTION RULES (NO HALLUCINATION):
1. **ONLY extract device names that ACTUALLY EXIST in the list above**
2. Convert ordinals: "2nd chiller" → "chiller 2"
3. Keep device type + number: "chiller 2", "SSB 3", "diff oil"
4. Remove keywords: report, performance, metrics, data, for, the, past, months
5. If "that device"/"it"/"same" → use conversation history
6. **If parameter query (e.g., "diff oil")** → IGNORE parameter, extract ONLY device:
   - "diff oil for chiller 2" → extracted_device: "chiller 2"
   - "supply temp of SSB 3" → extracted_device: "SSB 3"
   - "chiller 1 return pressure" → extracted_device: "chiller 1"

VALIDATION:
- Check if extracted device EXISTS in available list
- If not found → device_exists = false

EXAMPLES:
✅ Query: "diff oil for chiller 2", Available: [Chiller 1, Chiller 2, Chiller 3]
   → {{"extracted_device": "chiller 2", "device_exists": true}}

✅ Query: "5 months for chiller 2", Available: [Chiller 1, Chiller 2]
   → {{"extracted_device": "chiller 2", "device_exists": true}}

❌ Query: "chiller 4 performance", Available: [Chiller 1, Chiller 2, Chiller 3]
   → {{"extracted_device": null, "device_exists": false}}

✅ Query: "supply temp chiller 1", Available: [Chiller 1, Chiller 2]
   → {{"extracted_device": "chiller 1", "device_exists": true}}

Return JSON:
{{
    "extracted_device": "<device name that EXISTS>" or null,
    "device_exists": true or false,
    "confidence": <0.0 to 1.0>,
    "reasoning": "brief explanation"
}}
"""
    
    try:
        response1 = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": stage1_prompt}],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=150
        )
        
        stage1_result = json.loads(response1.choices[0].message.content)
        extracted_device = stage1_result.get('extracted_device', '').lower().strip() if stage1_result.get('extracted_device') else None
        device_exists = stage1_result.get('device_exists', False)
        stage1_confidence = stage1_result.get('confidence', 0.0)
        stage1_reasoning = stage1_result.get('reasoning', '')
        
        logging.info(f"✅ Stage 1 Result:")
        logging.info(f"   Extracted: '{extracted_device}'")
        logging.info(f"   Device Exists: {device_exists}")
        logging.info(f"   Confidence: {stage1_confidence:.2f}")
        logging.info(f"   Reasoning: {stage1_reasoning}")
        
        # ✅ NEW: Check if device actually exists
        if not extracted_device or not device_exists or stage1_confidence < 0.5:
            logging.warning(f"⚠️ Stage 1 failed - device doesn't exist or low confidence")
            logging.info(f"{'='*80}\n")
            return None, None, None, None, None
    
    except Exception as e:
        logging.error(f"❌ Stage 1 failed: {e}")
        return None, None, None, None, None
    
    # ==================== STAGE 2: MATCHING ====================
    logging.info(f"\n--- STAGE 2: Matching Against Database ---")
    
    stage2_prompt = f"""
You are a precise device matcher. Match the extracted device to the EXACT device in the database.

Extracted Device: "{extracted_device}"

Available Devices (ONLY these exist - COMPLETE LIST):
{json.dumps(device_list, indent=2)}

MATCHING RULES:
1. **ONLY match devices that ACTUALLY EXIST in the JSON list**
2. **Number matching MUST be exact**: 
   - "chiller 2" matches ONLY "Chiller 2"
   - NOT "Chiller 1" or "Chiller 3"
3. **Case-insensitive matching**
4. **Partial matching allowed for names**:
   - "ssb 3" matches "SSB 3-Chiller Panel"
   - "n-6 tank" matches "N-6 Cold Water Tank"

ANTI-HALLUCINATION:
❌ NEVER return device_id not in the JSON list
❌ NEVER match different numbers
❌ If requested device doesn't exist → device_id = null

Return JSON:
{{
    "device_id": <matched device_id or null>,
    "matched_name": "<exact device name from list>" or null,
    "confidence": <0.0 to 1.0>,
    "validation_passed": <true or false>
}}
"""    
    try:
        response2 = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": stage2_prompt}],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=200
        )
        stage2_result = json.loads(response2.choices[0].message.content)
        device_id = stage2_result.get('device_id')
        matched_name = stage2_result.get('matched_name')
        stage2_confidence = stage2_result.get('confidence', 0.0)
        validation_passed = stage2_result.get('validation_passed', True)
        
        logging.info(f"✅ Stage 2 Result:")
        logging.info(f"   Matched Device: {matched_name}")
        logging.info(f"   Device ID: {device_id}")
        logging.info(f"   Confidence: {stage2_confidence:.2f}")
        logging.info(f"   Validation: {'PASSED' if validation_passed else 'FAILED'}")
        
        # ✅ STRICT VALIDATION
        if device_id is not None and stage2_confidence > 0.75 and validation_passed:
            matched = df[df['device_id'] == device_id]
            
            if matched.empty:
                logging.error(f"❌ CRITICAL: Device ID {device_id} not in dataframe")
                return None, None, None, None, None
            
            row = matched.iloc[0]
            
            logging.info(f"\n✅ TWO-STAGE RESOLUTION SUCCESS:")
            logging.info(f"   Extracted: '{extracted_device}'")
            logging.info(f"   Matched: {row['device_name']} (ID: {device_id})")
            logging.info(f"   Confidence: {min(stage1_confidence, stage2_confidence):.2f}")
            logging.info(f"{'='*80}\n")
            
            return (
                int(row['device_id']),
                row['device_name'],
                int(row['device_type']),
                row['device_type_name'],
                row
            )
        
        logging.warning(f"⚠️ Stage 2 failed - no match")
        logging.info(f"{'='*80}\n")
        return None, None, None, None, None
        
    except Exception as e:
        logging.error(f"❌ Stage 2 failed: {e}")
        logging.info(f"{'='*80}\n")
        return None, None, None, None, None
    
def detect_list_devices_query(user_msg: str) -> bool:
    msg = user_msg.lower().strip()

    patterns = [
        "what are the devices",
        "list devices",
        "show devices",
        "devices in this vertical",
        "available devices",
        "how many devices",
        "list of device",
        "list of sensors",
        "device list",
        "what devices"
    ]

    return any(p in msg for p in patterns)



def resolve_device_from_df_regex(df: pd.DataFrame, query: str):
    """..."""
    if df.empty:
        return None, None, None, None, None
    
    query_clean = re.sub(r'[^a-z0-9]', '', query.lower())
    has_any_overlap = False
    
    for _, row in df.iterrows():
        device_clean = re.sub(r'[^a-z0-9]', '', row['device_name'].lower())
        # Check if query shares at least 3 characters or 40% overlap
        overlap = sum(1 for c in query_clean if c in device_clean)
        if overlap >= 3 or overlap / len(query_clean) > 0.4:
            has_any_overlap = True
            break
    
    if not has_any_overlap:
        logging.warning(f"⚠️  '{query}' has no character overlap with any device - SKIP LLM")
        return None, None, None, None, None
    
    logging.info(f"🤖 Using LLM for device resolution...")

    query_lower = query.strip().lower()

    # 1. Exact match
    exact = df[df['device_name'].str.lower() == query_lower]
    if not exact.empty:
        row = exact.iloc[0]
        return int(row['device_id']), row['device_name'], int(row['device_type']), row['device_type_name'], row

    # 2. Chiller number match
    chiller_match = re.search(r'chiller\s*(\d+)', query_lower)
    if chiller_match:
        chiller_num = chiller_match.group(1)
        chiller_name = f"chiller {chiller_num}"
        match = df[df['device_name'].str.lower() == chiller_name]
        if not match.empty:
            row = match.iloc[0]
            return int(row['device_id']), row['device_name'], int(row['device_type']), row['device_type_name'], row
        else:
            logging.warning(f"⚠️  Chiller {chiller_num} not found")
            return None, None, None, None, None

    # 3. Partial match
    for idx, row in df.iterrows():
        device_name_lower = row['device_name'].lower()
        if 'chiller' in device_name_lower:
            continue
        
        device_name_clean = re.sub(r'[^a-z0-9]', '', device_name_lower)
        query_clean = re.sub(r'[^a-z0-9]', '', query_lower)
        
        if device_name_clean in query_clean or device_name_lower in query_lower:
            return int(row['device_id']), row['device_name'], int(row['device_type']), row['device_type_name'], row

    return None, None, None, None, None

# ==================== LOGGING SETUP ====================
SCRIPT_DIR = Path(__file__).parent.resolve()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# ==================== MODULE INITIALIZATION ====================
app_start_time = time.time()
logging.info("="*80)
logging.info("CHILLER MONITORING CHATBOT - STARTING")
logging.info("="*80)

# --- Config ---
t = time.time()
logging.info(f"Config loaded in {time.time() - t:.2f}s")

# --- Memory ---
t = time.time()
from memory import (
    redis_client, generate_session_id, get_conversation_history,
    add_message_to_history, clear_conversation_history
)
logging.info(f"Memory module loaded in {time.time() - t:.2f}s")

# --- Database ---
t = time.time()
from database import validate_query_safety
logging.info(f"Database module loaded in {time.time() - t:.2f}s")

# --- RAG ---
t = time.time()
from rag_system import RAGSystem, auto_load_documents
logging.info(f"RAG module loaded in {time.time() - t:.2f}s")

# --- Utils ---
t = time.time()
from utils import generate_excel_report_in_memory, parse_recommendation_response
logging.info(f"Utils module loaded in {time.time() - t:.2f}s")

# --- Chatbot ---
t = time.time()
from chatbot import chatbot_graph, ChatbotState, classify_query_intent
import chatbot as chatbot_module
logging.info(f"Chatbot module loaded in {time.time() - t:.2f}s")

# --- Flask ---
t = time.time()
app = Flask(__name__)
app.secret_key = 'your-secret-key-here-change-this-in-production'
CORS(app)
logging.info(f"Flask initialized in {time.time() - t:.2f}s")

# --- RAG System ---
t = time.time()
rag_system = RAGSystem()
chatbot_module.rag_system = rag_system
logging.info(f"RAG system initialized in {time.time() - t:.2f}s")

# --- OpenAI ---
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
if openai_client:
    logging.info("OpenAI client ready")
else:
    logging.warning("OpenAI API key missing – LLM disabled")

auto_load_documents(rag_system)

logging.info("="*80)
logging.info(f"APPLICATION READY! ({time.time() - app_start_time:.2f}s)")
logging.info("="*80)

# ==================== /chat ROUTE ====================

@app.route('/chat', methods=['POST'])
def chat():
    start_time = time.time()
    data = request.get_json()
    user_message = data.get('message', '').strip()
    session_id = data.get('session_id')
    vertical_id = data.get('vertical_id')

    if not user_message:
        return jsonify({"response": "Please enter a message."}), 400
    
    if not vertical_id:
        logging.error("❌ No vertical_id in session")
        return jsonify({"response": "Session expired. Please login again."}), 401

    # === 1. LOAD DEVICES ===
    df_devices = load_devices_for_vertical(vertical_id)
    
    if df_devices.empty:
        logging.error(f"❌ NO DEVICES FOUND for vertical_id={vertical_id}")
        return jsonify({"response": "No devices found in this vertical."}), 400
    
    device_list_for_llm = df_devices[['device_id', 'device_name', 'device_type_name']].to_dict('records')

    # === 1.5 LOAD PARAMETERS ===
    try:
        df_params = VERTICAL_PARAM_CACHE.get(vertical_id, pd.DataFrame())
        if df_params.empty:
            df_params = load_parameters_for_vertical(vertical_id, df_devices)
        
        pin_context = ""
        if not df_params.empty:
            grouped = df_params.groupby("device_name")
            for dev_name, group in grouped:
                pin_context += f"\nDevice: {dev_name}\n"
                for _, row in group.iterrows():
                    pin_context += f"  - {row['parameter']} ({row['pin']})\n"
        logging.info(f"✅ Loaded {len(df_params)} pins/parameters for LLM (from cache)")
    except Exception as e:
        logging.error(f"❌ FAILED to load datastream parameters: {e}")
        df_params = pd.DataFrame()
        pin_context = ""

    # === 2. CHECK FOR EXIT COMMAND ===
    if check_exit_command(user_message):
        logging.info("🚪 User exited current query")
        return jsonify({
            "response": "Query cancelled. How else can I help you?",
            "session_id": session_id
        })
    
    # âœ… NEW: Check if message is TRULY time-only (no device, no parameter)
    from parameter_query import detect_time_only

    time_only_result = detect_time_only(user_message)

    if time_only_result and awaiting_device_input:
        # User provided time-only update → apply to pending query
        logging.info(f"   ⏱️ Time-only follow-up detected: {time_only_result}")
        pending_time_range = time_only_result
        
        # Return request for device again WITH new time
        device_list_msg = f"Got it - time range updated to {time_only_result['start_time']} to {time_only_result['end_time']}.\n\n"
        device_list_msg += "Now please specify which device:\n"
        device_list_msg += "\n".join(f"- {d['device_name']}" for d in device_list_for_llm)
        
        return jsonify({
            "response": device_list_msg,
            "session_id": session_id,
            "awaiting_device_input": True
        }), 200

    # === 2.5 CHECK FOR FOLLOW-BACK STATE IN CONVERSATION HISTORY ===
    if not session_id:
        session_id = generate_session_id()

    conversation_history = get_conversation_history(session_id)

    # ✅ ENHANCED: Extract FULL last parameter context (device + parameter + time)
    last_param_context = None
    for msg in reversed(conversation_history):
        if '[PARAM_CONTEXT]' in msg.get('content', ''):
            import re
            dev_id_match = re.search(r'device_id="(\d+)"', msg['content'])
            dev_name_match = re.search(r'device_name="([^"]*)"', msg['content'])
            cluster_match = re.search(r'cluster_id="(\d+)"', msg['content'])
            param_match = re.search(r'last_parameter="([^"]*)"', msg['content'])
            time_start_match = re.search(r'time_start="([^"]*)"', msg['content'])
            time_end_match = re.search(r'time_end="([^"]*)"', msg['content'])
            
            if dev_id_match and dev_name_match:
                last_param_context = {
                    'device_id': int(dev_id_match.group(1)),
                    'device_name': dev_name_match.group(1),
                    'cluster_id': int(cluster_match.group(1)) if cluster_match else None,
                    'parameter': param_match.group(1) if param_match and param_match.group(1) != 'None' else None,
                    'time_start': time_start_match.group(1) if time_start_match and time_start_match.group(1) != 'None' else None,
                    'time_end': time_end_match.group(1) if time_end_match and time_end_match.group(1) != 'None' else None
                }
                logging.info(f"💾 Found FULL param context: {last_param_context}")
                break        

    # ✅ NEW: Detect if user is asking for NEW parameter (not device name)
    awaiting_device_input = False
    pending_parameter_query = None
    pending_time_range = None

    for msg in reversed(conversation_history):
        content = msg.get('content', '')
        if '[FOLLOW_BACK_STATE]' in content and msg.get('role') == 'system':
            logging.info(f"🔄 Found follow-back marker in history")
            
            # Check if current message is a NEW parameter query
            param_keywords = [
                'what about', 'how about', 'show me', 'also show', 'give me',
                'frequency', 'voltage', 'current', 'power', 'temperature', 
                'pressure', 'flow', 'level', 'speed', 'amp', 'energy'
            ]
            
            is_new_param_query = any(kw in user_message.lower() for kw in param_keywords)
            
            if is_new_param_query:
                logging.info(f"🔄 Detected NEW parameter query → clearing follow-back marker")
                # Clear follow-back marker
                conversation_history = [m for m in conversation_history if '[FOLLOW_BACK_STATE]' not in m.get('content', '')]
                # Save cleaned history
                if redis_client and session_id:
                    try:
                        redis_client.delete(f"chat:{session_id}")
                        for msg in conversation_history:
                            add_message_to_history(session_id, msg['role'], msg['content'])
                    except Exception as e:
                        logging.error(f"Failed to clean history: {e}")
                break  # Don't activate follow-back mode
            
            # Extract follow-back state
            awaiting_device_input = True
            import re
            param_match = re.search(r'parameter="([^"]*)"', content)
            time_start_match = re.search(r'time_start="([^"]*)"', content)
            time_end_match = re.search(r'time_end="([^"]*)"', content)
            
            pending_parameter_query = param_match.group(1) if param_match and param_match.group(1) != 'None' else None
            start_time = time_start_match.group(1) if time_start_match and time_start_match.group(1) != 'None' else None
            end_time = time_end_match.group(1) if time_end_match and time_end_match.group(1) != 'None' else None
            
            if start_time and end_time:
                pending_time_range = {
                    'start_time': start_time,
                    'end_time': end_time,
                    'source': 'followback'
                }
            
            logging.info(f"   ✅ Restored: parameter='{pending_parameter_query}', time={pending_time_range}")
            break

    if awaiting_device_input:
        logging.info(f"\n🔄 FOLLOW-BACK MODE: User providing device name = '{user_message}'")
        
        # ✅ USE LLM for fuzzy device matching instead of exact match
        device_id, device_name, device_type, device_type_name, device_row = resolve_device_from_df_with_llm(
            df_devices, user_message, conversation_history
        )
        
        if not device_id:
            logging.warning(f"   ❌ Device not matched, asking again")
            # ... retry logic ...
            return jsonify({...}), 200
        
        # ✅ Device matched! Continue...
        logging.info(f"   ✅ Device matched: {device_name}")
        
        # Device matched! Proceed with parameter query using restored data
        is_parameter_query = True
        needs_device_resolution = False

        #✅ FIX: Extract cluster_id from device_row
        cluster_id = int(device_row['cluster_id']) if device_row is not None else None

        # ✅ CRITICAL: Ensure ALL device variables are set from matched device_row
        if device_row is not None:
            
            # âœ… CRITICAL FIX: Check if user is switching devices
            user_msg_lower = user_message.lower()
            preserve_parameter = False

            # Get previous device name from context
            previous_device_name = None
            for msg in reversed(conversation_history):
                if '[PARAM_CONTEXT]' in msg.get('content', ''):
                    import re
                    dev_name_match = re.search(r'device_name="([^"]*)"', msg['content'])
                    if dev_name_match:
                        previous_device_name = dev_name_match.group(1).lower()
                        break

            current_device_name = device_name.lower()

            # Check if device changed
            device_changed = previous_device_name and (previous_device_name != current_device_name)

            if device_changed:
                # Device changed → Clear parameter UNLESS user explicitly mentioned it
                if pending_parameter_query and pending_parameter_query.lower() in user_msg_lower:
                    preserved_parameter = pending_parameter_query
                    logging.info(f"„ Device changed BUT user mentioned '{pending_parameter_query}' → preserving")
                else:
                    preserved_parameter = None
                    logging.info(f"   🗑️ Device changed ({previous_device_name} → {current_device_name}) → clearing parameter")
            else:
                # Same device → Check if user wants same parameter
                if 'same' in user_msg_lower or 'also' in user_msg_lower:
                    preserve_parameter = True
                    logging.info(f"„ User said 'same'/'also' → preserving parameter")
                elif pending_parameter_query and pending_parameter_query.lower() in user_msg_lower:
                    preserve_parameter = True
                    logging.info(f"  User mentioned '{pending_parameter_query}' → preserving parameter")
                
                if preserve_parameter:
                    preserved_parameter = pending_parameter_query
                    logging.info(f"   💾 Preserved parameter: '{preserved_parameter}'")
                else:
                    preserved_parameter = None
                    logging.info(f"   🗑️ Cleared parameter (no explicit mention)")
            
            # ✅ ALWAYS preserve time from pending state
            preserved_time = pending_time_range
            logging.info(f"   💾 Preserved time: {preserved_time}")

            device_id = int(device_row['device_id'])
            device_name = device_row['device_name']
            device_type = int(device_row['device_type'])
            device_type_name = device_row['device_type_name']
            cluster_id = int(device_row['cluster_id'])
            slave_id = device_row['slave_id']
            
            logging.info(f"   ✅ ALL device vars set: ID={device_id}, Name={device_name}, Cluster={cluster_id}")
            
            # ✅ CRITICAL: Store in dict with cleared/preserved parameter
            followback_device_data = {
                'device_id': device_id,
                'device_name': device_name,
                'device_type': device_type,
                'device_type_name': device_type_name,
                'cluster_id': cluster_id,
                'cluster_name': device_row['cluster_name'],
                'slave_id': slave_id,
                'preserved_parameter': preserved_parameter,  # ← Use the CLEARED/PRESERVED value
                'preserved_time': preserved_time
            }
            logging.info(f"   💾 Stored in followback_device_data: param={preserved_parameter}")
            
            # ✅ PRESERVE parameter and time from pending state
            preserved_parameter = pending_parameter_query
            preserved_time = pending_time_range
            
            logging.info(f"   💾 Preserved: param='{preserved_parameter}', time={preserved_time}")

            # âœ… CRITICAL FIX: Also clear time if device OR parameter changed
            if device_changed or not preserved_parameter:
                # Don't preserve old time - force re-extraction from user query
                preserved_time = None
                logging.info(f"   🗑️ Cleared preserved time (device/param changed)")
            else:
                preserved_time = pending_time_range
                logging.info(f"   💾 Preserved time: {preserved_time}")

             #✅ CRITICAL: Store in a persistent dict that won't be lost
            followback_device_data = {
                'device_id': device_id,
                'device_name': device_name,
                'device_type': device_type,
                'device_type_name': device_type_name,
                'cluster_id': cluster_id,
                'cluster_name': device_row['cluster_name'],
                'slave_id': slave_id,
                'preserved_parameter': preserved_parameter,  # ← Use the logic-determined value
                'preserved_time': preserved_time  # ← Use the cleared/preserved value
            }
            logging.info(f"   💾 Stored in followback_device_data dict")
            # ✅ Store preserved data too
            followback_device_data['preserved_parameter'] = preserved_parameter
            followback_device_data['preserved_time'] = preserved_time

        # Mark that we've exited follow-back
        logging.info(f"   ✅ Follow-back resolved!")
        logging.info(f"   Device: {device_name}")
        logging.info(f"   Parameter: {pending_parameter_query}")
        logging.info(f"   Time: {pending_time_range}")
        logging.info(f"   Cluster: {cluster_id}")
        
        # Mark that we've exited follow-back by removing marker from conversation for next query
        #conversation_history = [msg for msg in conversation_history if '[FOLLOW_BACK_STATE]' not in msg.get('content', '')]
        # ✅ CRITICAL: Remove follow-back marker from Redis (not just local variable)
        logging.info(f"   🗑️ Removing follow-back marker from conversation history")
        
        # Filter out follow-back markers
        cleaned_history = [msg for msg in conversation_history if '[FOLLOW_BACK_STATE]' not in msg.get('content', '')]
        
        # Save cleaned history back to Redis
        if redis_client and session_id:
            try:
                redis_client.delete(f"chat:{session_id}")
                for msg in cleaned_history:
                    add_message_to_history(session_id, msg['role'], msg['content'])
                logging.info(f"   ✅ Follow-back marker removed from Redis")
            except Exception as e:
                logging.error(f"   ❌ Failed to clean history: {e}")
        
        # Also update local variable
        conversation_history = cleaned_history
        
        # ✅ Clear follow-back state flags
        awaiting_device_input = False
        pending_parameter_query = None
        pending_time_range = None
        
        logging.info(f"   ✅ Follow-back state cleared")

        # ✅ CRITICAL: Keep device variables set - do NOT let them be overwritten later
        logging.info(f"   🔒 Locking device context: {device_name} (ID: {device_id})")
        
        # Set flag to prevent fallback override
        device_locked = True

        logging.info(f"   ✅ Follow-back resolved!")
        logging.info(f"   Device: {device_name}")
        logging.info(f"   Parameter: {followback_device_data.get('preserved_parameter')}")
        logging.info(f"   Time: {followback_device_data.get('preserved_time')}")

    # === 3. DETECT QUERY TYPE (EARLY - BEFORE device resolution) ===
    query_lower = user_message.lower()

    def detect_greeting_or_intro(user_msg: str) -> tuple:
        """Use LLM to detect greetings/intros"""
        if not openai_client:
            return False, False
        
        try:
            device_names = df_devices['device_name'].tolist() if not df_devices.empty else []
        except Exception:
            device_names = []

        device_context = ""
        if device_names:
            device_context = "Available device names:\n" + "\n".join(f"- {n}" for n in device_names[:200]) + "\n\n"

        prompt = f"""
Classify the user's message:
1. Greeting / casual / farewell
2. Name introduction
3. Not greeting

Device names (for disambiguation):
{device_context}

Message: "{user_msg}"

Return JSON:
{{
    "is_greeting": true/false,
    "is_name_intro": true/false,
    "confidence": <0.0-1.0>
}}
"""
        try:
            response = openai_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0,
                max_tokens=120
            )
            result = json.loads(response.choices[0].message.content)
            is_greeting = result.get('is_greeting', False) and result.get('confidence', 0) > 0.7
            is_intro = result.get('is_name_intro', False) and result.get('confidence', 0) > 0.7
            return is_greeting, is_intro
        except Exception as e:
            logging.error(f"❌ Greeting detection failed: {e}")
            return False, False

    is_pure_greeting, is_name_intro = detect_greeting_or_intro(user_message)
    needs_device_resolution = False
    # ✅ CHECK METRICS/REPORT/ADVISORY **BEFORE** parameter detection
    is_metrics = any(kw in query_lower for kw in ['cop', 'eer', 'performance', 'efficiency', 'perform', 'result', 'metrics'])
    is_report = any(kw in query_lower for kw in ['report', 'excel', 'download', 'xlsx', 'logs', 'loggs'])
    is_advisory = any(kw in query_lower for kw in ['how to', 'improve', 'optimize', 'maintain', 'advice', 'tips', 'reduce'])

    # Only check parameter query if NOT metrics/report/advisory
    is_parameter_query = False
    param_detection = {}
    
    if not is_metrics and not is_report and not is_advisory:
        device_list_for_llm = df_devices[['device_id', 'device_name', 'device_type_name']].to_dict('records')
        param_detection = detect_parameter_query(
            user_message,
            device_list=device_list_for_llm,
            parameter_list=df_params.to_dict("records") if not df_params.empty else None,
            pin_context=pin_context
        )
        is_parameter_query = param_detection.get('is_parameter_query', False) and param_detection.get('confidence', 0) > 0.7

    # ✅ NEW: Check for list_parameters query
    is_list_parameters = detect_list_parameters_query(user_message)
    
    # ✅ NEW: Check for list_devices query
    is_list_devices = detect_list_devices_query(user_message)

    logging.info(f"QUERY TYPE DETECTION:")
    logging.info(f"  Pure greeting: {is_pure_greeting}")
    logging.info(f"  Name intro: {is_name_intro}")
    logging.info(f"  Advisory: {is_advisory}")
    logging.info(f"  Parameter query: {is_parameter_query}")
    logging.info(f"  List parameters: {is_list_parameters}")
    logging.info(f"  List devices: {is_list_devices}")
    logging.info(f"  Awaiting device input (follow-back): {awaiting_device_input}")
    logging.info(f"  Needs device resolution: {needs_device_resolution}")

    # ✅ ADD THIS - Clear memory on intent switch
    if not is_parameter_query and (is_pure_greeting or is_advisory or is_list_parameters or is_list_devices):
        logging.info(f"🗑️ Clearing parameter context - intent switched")
        
        # Remove context markers
        conversation_history = [
            msg for msg in conversation_history 
            if '[PARAM_CONTEXT]' not in msg.get('content', '') and 
               '[FOLLOW_BACK_STATE]' not in msg.get('content', '')
        ]
        awaiting_device_input = False
        pending_parameter_query = None
        pending_time_range = None
        last_param_context = None
    # ✅ END ADD

    # ✅ FIX: Clear stale param context if last query was NOT a parameter query
    if last_param_context and not is_parameter_query:
        # Check if previous message was a successful parameter query
        last_was_param_query = False
        for msg in reversed(conversation_history):
            if msg.get('role') == 'assistant':
                content = msg.get('content', '')
                # Check if it's a parameter result (has statistics format)
                if 'Average:' in content or 'Minimum:' in content or 'Data Points:' in content:
                    last_was_param_query = True
                break
        
        # If last wasn't param query, clear context
        if not last_was_param_query:
            logging.info(f"🗑️ Clearing stale param context (last query was not param query)")
            last_param_context = None
    
    # ✅ CHECK: Is this a follow-up parameter query? (before follow-back check)
    if is_parameter_query and last_param_context and not awaiting_device_input:
        query_lower = user_message.lower()
        followup_indicators = ['what about', 'how about', 'show me', 'and', 'also', 'now', 'give me']
        
        # More strict: Must have follow-up phrase AND parameter keyword
        has_followup_phrase = any(ind in query_lower for ind in followup_indicators)
        
        # Check if device is NOT mentioned (to avoid "what about chiller 2?" being treated as follow-up)
        device_names_lower = [d['device_name'].lower() for d in device_list_for_llm]
        device_mentioned = any(dn in query_lower for dn in device_names_lower)
        
        is_explicit_followup = has_followup_phrase and not device_mentioned
        
        if is_explicit_followup:
            logging.info(f"🔄 Follow-up parameter query detected - reusing device context")
            device_id = last_param_context['device_id']
            device_name = last_param_context['device_name']
            cluster_id = last_param_context['cluster_id']
            
            # Get full device row
            device_row = df_devices[df_devices['device_id'] == device_id].iloc[0]
            device_type = int(device_row['device_type'])
            device_type_name = device_row['device_type_name']
            needs_device_resolution = False
            
            logging.info(f"   ✅ Reusing device: {device_name} (ID: {device_id})")


    # ✅ CRITICAL: If follow-back is active, check what the user said
    if awaiting_device_input:
        logging.info(f"\n🔄 FOLLOW-BACK MODE ACTIVE - Checking user response")
        
        # Check if user is providing a device name (not a new query type)
        user_msg_lower = user_message.lower().strip()
        device_names_lower = [d.lower() for d in df_devices['device_name'].tolist()]
        
        # Is the user's message a device name? (check if we successfully matched earlier)
        is_device_name_response = 'device_locked' in locals() and locals().get('device_locked', False)
        
        # Is the user asking for something NEW (new intent)?
        is_new_intent = is_list_parameters or is_list_devices or is_advisory or is_pure_greeting
        
        logging.info(f"   User message: '{user_message}'")
        logging.info(f"   Is device name? {is_device_name_response}")
        logging.info(f"   Is new intent? {is_new_intent} (list_params={is_list_parameters}, list_devices={is_list_devices}, advisory={is_advisory}, greeting={is_pure_greeting})")
        
        # If user said something NEW (not a device name), CLEAR follow-back
        if is_new_intent and not is_device_name_response:
            logging.info(f"🔄 NEW INTENT DETECTED → CLEARING follow-back marker")
            
            # Remove follow-back marker from conversation history
            conversation_history = [
                msg for msg in conversation_history 
                if '[FOLLOW_BACK_STATE]' not in msg.get('content', '')
            ]
            
            # Clear follow-back state
            awaiting_device_input = False
            pending_parameter_query = None
            pending_time_range = None
            
            logging.info(f"✅ Follow-back marker CLEARED → treating as NEW INDEPENDENT query")

    # ✅ Handle list_parameters query
    if is_list_parameters:
        logging.info(f"📋 LIST PARAMETERS query detected")
        # Will be handled through normal chatbot flow
        needs_device_resolution = True

    # ✅ Handle list_devices query
    if is_list_devices:
        logging.info(f"📱 LIST DEVICES query detected")
        device_names = df_devices['device_name'].tolist()
        response_text = "Devices in this vertical:\n\n" + "\n".join(f"- {d}" for d in device_names)
        return jsonify({
            "response": response_text,
            "session_id": session_id
        }), 200

    needs_device_resolution = not is_pure_greeting and not is_name_intro and not is_advisory

    # === 3.5 SPECIAL: Parameter Query Follow-back (ONLY if still in parameter_query) ===
    if is_parameter_query and awaiting_device_input:
        logging.info(f"🔄 PARAMETER QUERY with follow-back active - will use restored device/time")

    # ✅ ADD THIS - Check for follow-up parameter query
    device_id = None  # Initialize device_id first
    if is_parameter_query and not device_id and last_param_context:
        query_lower = user_message.lower()
        followup_indicators = ['what about', 'how about', 'show me', 'and', 'also', 'now']
        # Only reuse context if it's EXPLICIT follow-up phrase
        # NOT if it's a new parameter query without device
        is_explicit_followup = any(ind in query_lower for ind in followup_indicators)
        
        if is_explicit_followup:
            logging.info(f"🔄 Follow-up parameter query - reusing device context")
            device_id = last_param_context['device_id']
            device_name = last_param_context['device_name']
            cluster_id = last_param_context['cluster_id']
            device_row = df_devices[df_devices['device_id'] == device_id].iloc[0]
            device_type = int(device_row['device_type'])
            device_type_name = device_row['device_type_name']
            needs_device_resolution = False
    # ✅ END ADD    

    # === 3.6 SPECIAL: Follow-up Parameter Query (if device not yet resolved) ===
    if is_parameter_query and not device_id and last_param_context:
        query_lower = user_message.lower()
        # Keywords indicating a follow-up on the same device
        followup_indicators = ['what about', 'how about', 'show me', 'and', 'also', 'now', 'for', 'same time']
        
        # Check if any device name is mentioned. If so, it's NOT a follow-up for the *previous* device.
        device_names_lower = [d['device_name'].lower() for d in device_list_for_llm]
        device_mentioned = any(dn in query_lower for dn in device_names_lower)

        is_explicit_followup = any(ind in query_lower for ind in followup_indicators)

        # A query is a follow-up if it has follow-up words AND does NOT mention a new device.
        if is_explicit_followup and not device_mentioned:
            logging.info(f"🔄 Follow-up parameter query detected - reusing previous device context")
            device_id = last_param_context['device_id']
            device_name = last_param_context['device_name']
            cluster_id = last_param_context['cluster_id']
            
            # Get the full device row from the dataframe
            device_row_match = df_devices[df_devices['device_id'] == device_id]
            if not device_row_match.empty:
                device_row = device_row_match.iloc[0]
                device_type = int(device_row['device_type'])
                device_type_name = device_row['device_type_name']
                needs_device_resolution = False # We have our device!
                logging.info(f"   ✅ Reusing Device: {device_name} (ID: {device_id}), Cluster: {cluster_id}")

    # === 4. DEVICE RESOLUTION ===
    if needs_device_resolution and not awaiting_device_input:
        logging.info(f"\n🔍 RESOLVING DEVICE USING LLM: '{user_message}'")
        
        device_id, device_name, device_type, device_type_name, device_row = resolve_device_from_df_with_llm(
            df_devices, user_message, conversation_history
        )

        if not device_id:
            logging.error(f"❌ Device resolution failed for query: '{user_message}'")
            
            if is_parameter_query:
                from parameter_query import get_available_devices_message
                available_msg = get_available_devices_message(df_devices)
                
                logging.warning(f"❌ DEVICE NOT FOUND for parameter query")
                
                # ✅ Extract time from the ORIGINAL query (not from param_detection which doesn't have it)
                time_prompt = f"""
        Extract time range from: "{user_message}"
        Current Date: {datetime.now().strftime('%Y-%m-%d')}

        Return JSON:
        {{
            "start_time": "YYYY-MM-DDTHH:MM:SS" or null,
            "end_time": "YYYY-MM-DDTHH:MM:SS" or null,
            "source": "explicit" or "none"
        }}
        """
                time_info = {}
                try:
                    time_response = openai_client.chat.completions.create(
                        model=OPENAI_MODEL,
                        messages=[{"role": "user", "content": time_prompt}],
                        response_format={"type": "json_object"},
                        temperature=0,
                        max_tokens=150
                    )
                    time_data = json.loads(time_response.choices[0].message.content)
                    time_info = {
                        'start_time': time_data.get('start_time'),
                        'end_time': time_data.get('end_time'),
                        'source': time_data.get('source', 'none')
                    }
                    logging.info(f"   ✅ Extracted time: {time_info}")
                except Exception as e:
                    logging.error(f"   ⚠️  Time extraction failed: {e}")
                
                # Store follow-back state with BOTH parameter AND time
                start_time = time_info.get('start_time') if time_info else None
                end_time = time_info.get('end_time') if time_info else None
                
                followback_marker = {
                    'role': 'system',
                    'content': f'[FOLLOW_BACK_STATE] parameter="{param_detection.get("parameter_name")}" time_start="{start_time}" time_end="{end_time}"'
                }
                add_message_to_history(session_id, followback_marker['role'], followback_marker['content'])
                logging.info(f"✅ Stored follow-back marker: {followback_marker['content']}")
                
                return jsonify({
                    "response": available_msg,
                    "session_id": session_id,
                    "awaiting_device_input": True,
                    "pending_query": user_message
                }), 200

            device_names = df_devices['device_name'].tolist()[:15]
            available_msg = f"Device not found. Available devices:\n{', '.join(device_names)}"
            
            logging.error(f"❌ DEVICE NOT FOUND")
            return jsonify({"response": available_msg}), 400

        logging.info(f"✅ RESOLVED DEVICE:")
        logging.info(f"   Name: {device_name}")
        logging.info(f"   Type: {device_type_name} (ID: {device_type})")
        logging.info(f"   Device ID: {device_id}")
        logging.info("="*80)
    

    # Fallback for greetings (but NOT if we're in parameter query mode)
    if not device_id and not is_parameter_query and not locals().get('device_locked', False):
        logging.info(f"👋 GREETING/INTRO - Using placeholder device")
        row = df_devices.iloc[0]
        device_id = int(row['device_id'])
        device_name = row['device_name']
        device_type = int(row['device_type'])
        device_type_name = row['device_type_name']
        device_row = row
    elif not device_id and is_parameter_query:
        # Should not happen - parameter query must have device by now
        logging.error(f"❌ CRITICAL: Parameter query without device!")

    # === 5. PARAMETER DF + PIN CONTEXT ===
    try:
        sql = """
        SELECT d.device_id, d.device_name, p.pin, p.parameter_name AS parameter
        FROM parameters p
        JOIN devices d ON p.device_id = d.device_id
        WHERE d.device_id IN :device_ids AND p.isdelete = 0
        """
        with engine.connect() as conn:
            df_params = pd.read_sql(
                text(sql),
                conn,
                params={"device_ids": tuple(df_devices["device_id"].unique())}
            )
    except:
        df_params = pd.DataFrame()

    pin_context = "Available Parameters:\n"
    if not df_params.empty:
        for _, row in df_params.iterrows():
            pin_context += f"- {row['parameter']} ({row['pin']}) [{row['device_name']}]\n"
    pin_context += "\n"

    # === 6. CREATE CHATBOT STATE ===
    if not session_id:
        session_id = generate_session_id()

    conversation_history = get_conversation_history(session_id)

    # ✅ DEBUG: Check what we're passing to ChatbotState
    logging.info(f"\n{'='*80}")
    logging.info(f"📦 PRE-STATE CHECK:")
    logging.info(f"   device_id exists? {device_id if 'device_id' in locals() else 'NOT DEFINED'}")
    logging.info(f"   followback_device_data exists? {'YES' if 'followback_device_data' in locals() else 'NO'}")
    if 'followback_device_data' in locals():
        logging.info(f"   followback_device_data: {followback_device_data}")
    logging.info(f"{'='*80}\n")

    # ✅ SELECTIVE MEMORY: Build intent_data with smart fallback logic
    # 1. Device resolution
    final_device_id = None
    final_device_name = None
    final_cluster_id = None
    final_device_type = None
    final_device_type_name = None
    final_slave_id = None
    if 'followback_device_data' in locals():
        # Use followback device (user just provided device name)
        final_device_id = followback_device_data['device_id']
        final_device_name = followback_device_data['device_name']
        final_cluster_id = followback_device_data.get('cluster_id')
        final_device_type = followback_device_data.get('device_type')
        final_device_type_name = followback_device_data.get('device_type_name')
        final_slave_id = followback_device_data.get('slave_id')
        logging.info(f"   📍 Using NEW device: {final_device_name}")
    elif device_id:
        # Use resolved device from query
        final_device_id = device_id
        final_device_name = device_name
        final_cluster_id = cluster_id if 'cluster_id' in locals() else (device_row['cluster_id'] if device_row is not None else None)
        final_device_type = device_type if 'device_type' in locals() else None
        final_device_type_name = device_type_name if 'device_type_name' in locals() else None
        final_slave_id = device_row['slave_id'] if device_row is not None else None
        logging.info(f"   📍 Using RESOLVED device: {final_device_name}")    
    elif last_param_context and last_param_context.get('device_id'):
        # Fallback to previous device from memory
        final_device_id = last_param_context['device_id']
        final_device_name = last_param_context['device_name']
        final_cluster_id = last_param_context.get('cluster_id')
        logging.info(f"   📍 Using PREVIOUS device: {final_device_name}")

    # 2. Parameter resolution
    final_parameter = None    
    
    if 'followback_device_data' in locals() and followback_device_data.get('preserved_parameter'):
        # Use preserved parameter from followback
        final_parameter = followback_device_data['preserved_parameter']
        logging.info(f"   📍 Using PRESERVED parameter: {final_parameter}")
    elif is_parameter_query and param_detection.get('parameter_name'):
        # Use newly detected parameter
        final_parameter = param_detection.get('parameter_name')
        logging.info(f"   📍 Using NEW parameter: {final_parameter}")
    elif pending_parameter_query:
        # Use pending parameter from follow-back
        final_parameter = pending_parameter_query
        logging.info(f"   📍 Using PENDING parameter: {final_parameter}") 
    elif last_param_context and last_param_context.get('parameter'):
        # Fallback to previous parameter from memory
        final_parameter = last_param_context['parameter']
        logging.info(f"   📍 Using PREVIOUS parameter: {final_parameter}")    

    # 3. Time resolution
    final_time_info = None           

    if 'followback_device_data' in locals() and followback_device_data.get('preserved_time'):
        # Use preserved time from followback
        final_time_info = followback_device_data['preserved_time']
        logging.info(f"   📍 Using PRESERVED time: {final_time_info}")
    elif pending_time_range:
        # Use pending time from follow-back
        final_time_info = pending_time_range
        logging.info(f"   📍 Using PENDING time: {final_time_info}")   
    else:
        # Extract time from current query OR use previous time
        from datetime import datetime as dt
        current_time = dt.now()     

        time_prompt = f"""
Extract time range from query OR use previous time.

Current Query: "{user_message}"
Current Date: {current_time.strftime('%Y-%m-%d')}
Previous Time: {last_param_context.get('time_start') if last_param_context else 'None'} to {last_param_context.get('time_end') if last_param_context else 'None'}

RULES:
1. If query has time words (past, months, days, june, july) â†' extract NEW time
2. If NO time words AND previous time exists â†' USE previous time
3. If NO time anywhere â†' return null

Return JSON:
{{
    "start_time": "YYYY-MM-DDTHH:MM:SS" or null,
    "end_time": "YYYY-MM-DDTHH:MM:SS" or null,
    "source": "explicit" or "history" or "none"
}}
"""
        try:
            time_response = openai_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": time_prompt}],
                response_format={"type": "json_object"},
                temperature=0,
                max_tokens=200
            )
            time_data = json.loads(time_response.choices[0].message.content)
            
            start_time = time_data.get('start_time')
            end_time = time_data.get('end_time')
            time_source = time_data.get('source', 'none')

            if start_time and end_time:
                final_time_info = {
                    'start_time': start_time,
                    'end_time': end_time,
                    'source': time_source
                }
                logging.info(f"   📍 Using NEW time: {final_time_info}")
            elif last_param_context and last_param_context.get('time_start'):
                # Fallback to previous time
                final_time_info = {
                    'start_time': last_param_context['time_start'],
                    'end_time': last_param_context['time_end'],
                    'source': 'history'
                }
                logging.info(f"   📍 Using PREVIOUS time: {final_time_info}")    
        except Exception as e:
            logging.error(f"   ❌ Time extraction failed: {e}")
            if last_param_context and last_param_context.get('time_start'):
                final_time_info = {
                    'start_time': last_param_context['time_start'],
                    'end_time': last_param_context['time_end'],
                    'source': 'history'
                }
                logging.info(f"   📍 Using PREVIOUS time (fallback): {final_time_info}")  

    # Build final intent_data
    initial_state = ChatbotState(
        user_message=user_message,
        session_id=session_id,
        conversation_history=conversation_history,
        device_list_for_llm=device_list_for_llm,
        parameter_list_for_llm=df_params.to_dict("records") if not df_params.empty else [],
        pin_context_for_llm=pin_context,        

        intent_data={
            'device_id': final_device_id,
            'device_name': final_device_name,
            'device_type': final_device_type,
            'device_type_name': final_device_type_name,
            'cluster_id': final_cluster_id,
            'slave_id': final_slave_id,
            'parameter_name': final_parameter,
            'time_info': final_time_info or {}
        },          

        awaiting_device_input=awaiting_device_input,
        pending_parameter_query=pending_parameter_query if awaiting_device_input else None,
        pending_time_range=pending_time_range if awaiting_device_input else None,
        pending_device_id=None,

        response_data={},
        final_response="",
        response_time=0.0,
        start_time=datetime.now(),
    )

    logging.info(f"\n{'='*80}")
    logging.info(f"📦 FINAL STATE CHECK:")
    logging.info(f"   Device: {final_device_name} (ID: {final_device_id})")
    logging.info(f"   Parameter: {final_parameter}")
    logging.info(f"   Time: {final_time_info}")
    logging.info(f"{'='*80}\n")
    
    # === 7. EXECUTE CHATBOT ===
    result = chatbot_graph.invoke(initial_state)
    response = result['final_response']
    add_message_to_history(session_id, "assistant", response)

    logging.info(f"\n✅ CHATBOT COMPLETE")
    logging.info(f"Response: {response[:200]}...")
    logging.info(f"Time: {result['response_time']:.2f}s")
    logging.info("="*80 + "\n")

    response_payload = {
        "response": response,
        "session_id": session_id,
        "response_time": result["response_time"]
    }
    if result.get('response_data', {}).get('download_ready'):
        response_payload['download_ready'] = True
        response_payload['report_id'] = result['response_data']['report_id']
        response_payload['download_filename'] = result['response_data']['download_filename']
        response_payload['rows'] = result['response_data']['rows']
        response_payload['file_size'] = result['response_data']['file_size']
        logging.info(f"📦 Download ready: {response_payload['report_id']}")

    return jsonify(response_payload)


# ==================== ALL OTHER ROUTES (UNCHANGED) ====================

@app.route('/chat/clear', methods=['POST'])
def clear_chat():
    data = request.get_json()
    session_id = data.get('session_id')
    logging.info(f"Clear history: {session_id[:8] if session_id else 'None'}")
    if not session_id:
        return jsonify({"error": "session_id required"}), 400
    success = clear_conversation_history(session_id)
    return jsonify({"message": "Cleared", "session_id": session_id}) if success else jsonify({"error": "Failed"}), 500

@app.route('/download_excel/<report_id>', methods=['GET'])
def download_excel(report_id):
    logging.info(f"Download: {report_id}")
    if redis_client:
        data = redis_client.get(f"excel:{report_id}")
        if data:
            excel_bytes = base64.b64decode(data)
            filename = request.args.get('filename', f'{report_id}.xlsx')
            return Response(
                excel_bytes,
                mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                headers={'Content-Disposition': f'attachment; filename={filename}'}
            )
    return jsonify({"error": "Not found"}), 404

@app.route('/upload_document', methods=['POST'])
def upload_document():
    if 'file' not in request.files:
        return jsonify({"error": "No file"}), 400
    file = request.files['file']
    if not file.filename:
        return jsonify({"error": "No file selected"}), 400
    ext = Path(file.filename).suffix.lower()
    if ext not in {'.txt', '.pdf'}:
        return jsonify({"error": "Invalid file"}), 400
    upload_path = SCRIPT_DIR / "uploads" / file.filename
    upload_path.parent.mkdir(exist_ok=True)
    file.save(str(upload_path))
    success = rag_system.add_document(str(upload_path))
    upload_path.unlink()
    return jsonify({"message": "Success", "chunks": len(rag_system.chunks)}) if success else jsonify({"error": "Failed"}), 500

@app.route('/health')
def health():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "rag_chunks": len(rag_system.chunks),
        "redis": bool(redis_client)
    }), 200

@app.route('/')
def index():
    return jsonify({
        "message": "IoT Devices Monitoring Assistant API",
        "version": "1.0.0",
        "endpoints": {"/chat": "POST", "/chat/clear": "POST", "/download_excel/<id>": "GET"}
    })

@app.route('/load_devices', methods=['POST'])
def load_devices():
    data = request.get_json()
    vertical_id = data.get('vertical_id')
    
    if not vertical_id:
        return jsonify({"error": "vertical_id required"}), 400

    logging.info("\n" + "="*80)
    logging.info(f"LOAD DEVICES FOR VERTICAL ID: {vertical_id}")
    logging.info("="*80)

    df_devices = load_devices_for_vertical(vertical_id)
    
    if df_devices.empty:
        logging.warning(f"NO DEVICES for vertical_id={vertical_id}")
        return jsonify({"devices": []}), 200

    devices_list = df_devices[[
        'device_id', 'device_name', 'device_type', 'device_type_name', 'cluster_name'
    ]].to_dict('records')

    logging.info(f"RETURNED {len(devices_list)} devices to frontend")
    return jsonify({"devices": devices_list}), 200

@app.route('/clear_cache', methods=['POST'])
def clear_cache():
    data = request.get_json()
    vertical_id = data.get('vertical_id')
    
    with CACHE_LOCK:
        if vertical_id in VERTICAL_DEVICE_CACHE:
            del VERTICAL_DEVICE_CACHE[vertical_id]
            logging.info(f"🗑️  CLEARED CACHE for vertical_id={vertical_id}")
            return jsonify({"message": "Cache cleared"}), 200
    
    return jsonify({"message": "No cache found"}), 200

# ==================== LOGIN ROUTE ====================
from flask import request, jsonify
from sqlalchemy import text
import json
import logging
import pandas as pd

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json() or {}
    username = data.get('username', '').strip()
    password = data.get('password', '').strip()
    
    if not username or not password:
        return jsonify({"success": False, "message": "Username and password required"}), 400

    # Authenticate user
    user_data = authenticate_user(username, password)
    if not user_data:
        logging.warning(f"❌ Login failed: {username}")
        return jsonify({"success": False, "message": "Invalid username or password"}), 401

    # At this point user authenticated
    try:
        vertical_id = int(user_data['vertical_id'])
    except Exception as e:
        logging.error(f"❌ Invalid vertical_id from user_data: {e}")
        return jsonify({"success": False, "message": "Server configuration error"}), 500

    logging.info(f"✅ Login successful: {username} → {user_data.get('vertical_name')} (vertical_id={vertical_id})")

    # 1) Load devices for this vertical (should return a DataFrame with device_id, device_name, cluster_id, etc.)
    try:
        df_devices_local = load_devices_for_vertical(vertical_id)
        # Ensure device_id column exists and numeric
        if df_devices_local is None or df_devices_local.empty:
            logging.warning(f"⚠️ No devices found for vertical {vertical_id}")
            df_devices_local = pd.DataFrame(columns=["device_id", "device_name", "cluster_id"])
    except Exception as e:
        logging.error(f"❌ Failed to load devices for vertical {vertical_id}: {e}")
        df_devices_local = pd.DataFrame(columns=["device_id", "device_name", "cluster_id"])

    try:
        # Get cluster ids for vertical (cluster table columns: cluster_id, vertical_Id, isdelete)
        with engine.connect() as conn:
            cluster_rows = conn.execute(
                text("SELECT cluster_id FROM cluster WHERE vertical_Id = :vid AND isdelete = 0"),
                {"vid": vertical_id}
            ).fetchall()
        cluster_ids = [int(r[0]) for r in cluster_rows] if cluster_rows else []

        if not cluster_ids:
            logging.info(f"🔍 No clusters found for vertical {vertical_id}, caching empty params")
            PARAM_CACHE[vertical_id] = pd.DataFrame(columns=["device_id", "device_name", "pins", "parameter"])
        else:
            # Build safe parameterized IN clause placeholders
            placeholders = []
            params = {}
            for i, cid in enumerate(cluster_ids):
                key = f"c{i}"
                placeholders.append(f":{key}")
                params[key] = cid
            in_clause = ", ".join(placeholders)

            sql = text(f"""
                SELECT cluster_Id, address
                FROM datastream
                WHERE cluster_Id IN ({in_clause}) AND isdelete = 0
            """)

            with engine.connect() as conn:
                ds_rows = conn.execute(sql, params).fetchall()

            # ==================== DEBUG: Add this AFTER ds_rows = conn.execute(...) ====================
            # Place this code right after fetching ds_rows to see what's in the database

            logging.info(f"\n{'='*80}")
            logging.info(f"🔍 DEBUG: RAW DATASTREAM INSPECTION for vertical {vertical_id}")
            logging.info(f"{'='*80}")

            for idx, row in enumerate(ds_rows):
                cluster_id_val = row[0]
                address_json = row[1]
                
                logging.info(f"\n[Row {idx}] Cluster {cluster_id_val}:")
                logging.info(f"   Raw JSON length: {len(address_json) if address_json else 0}")
                
                if not address_json:
                    logging.info(f"   ⚠️  Empty address JSON")
                    continue
                
                try:
                    address_arr = json.loads(address_json)
                    logging.info(f"   Parsed as: {type(address_arr).__name__}")
                    
                    if isinstance(address_arr, list):
                        logging.info(f"   Array length: {len(address_arr)}")
                        
                        for block_idx, block in enumerate(address_arr):
                            if isinstance(block, dict):
                                pin = block.get("pin", "NO_PIN")
                                address_keys = [k for k in block.keys() if k.startswith("address-")]
                                logging.info(f"      Block[{block_idx}]: pin={pin}, address_keys={address_keys}")
                                
                                # Check for address data
                                for addr_key in address_keys:
                                    addr_data = block[addr_key]
                                    if isinstance(addr_data, dict):
                                        params = addr_data.get("params", "NO_PARAMS")
                                        devices = addr_data.get("devices", [])
                                        logging.info(f"         {addr_key}: params='{params}', devices={devices}")
                    else:
                        logging.info(f"   NOT an array! Type: {type(address_arr)}")
                except Exception as e:
                    logging.error(f"   ❌ Parse error: {e}")

            logging.info(f"{'='*80}\n")

            all_mappings = []
            # Build device lookup by id for quick membership tests
            if not df_devices_local.empty and "device_id" in df_devices_local.columns:
                device_lookup = df_devices_local.set_index("device_id")["device_name"].to_dict()
            else:
                device_lookup = {}

            for row in ds_rows:
                try:
                    cluster_id_val = row[0]
                    address_json = row[1]
                except Exception:
                    continue

                if not address_json:
                    continue

                # address_json expected to be a JSON array (list)
                try:
                    address_arr = json.loads(address_json)
                except Exception as e:
                    logging.debug(f"⚠️ Failed to parse address JSON: {e}")
                    continue

                if not isinstance(address_arr, list):
                    continue

                # Each block has a pin and address-1/address-2...
                for block in address_arr:
                    if not isinstance(block, dict):
                        continue
                    pin = block.get("pin")
                    if not pin:
                        continue

                    # iterate through address-keys dynamically
                    for sec_key, sec_val in block.items():
                        if not isinstance(sec_key, str) or not sec_key.startswith("address-"):
                            continue
                        if not isinstance(sec_val, dict):
                            continue

                        params_name = sec_val.get("params")
                        devices_list = sec_val.get("devices") or []

                        if not params_name or not isinstance(devices_list, list):
                            continue

                        for dev_id in devices_list:
                            try:
                                dev_id_int = int(dev_id)
                            except Exception:
                                continue
                            # only include if device exists in current vertical devices list
                            if dev_id_int in device_lookup:
                                all_mappings.append({
                                    "device_id": dev_id_int,
                                    "device_name": device_lookup[dev_id_int],
                                    "pin": pin,
                                    "parameter": params_name.strip() if isinstance(params_name, str) else params_name
                                })

            # Create DataFrame and remove duplicates
            if all_mappings:
                df_params_local = pd.DataFrame(all_mappings)
                # optional: normalize whitespace and remove duplicate rows
                df_params_local["parameter"] = df_params_local["parameter"].astype(str).str.strip()
                df_params_local = df_params_local.drop_duplicates(subset=['device_id', 'pin', 'parameter'], keep='first').reset_index(drop=True)
            else:
                df_params_local = pd.DataFrame(columns=["device_id", "device_name", "pin", "parameter"])

            # Cache the parameters
            PARAM_CACHE[vertical_id] = df_params_local
            logging.info(f"✅ Loaded {len(df_params_local)} parameter mappings for vertical {vertical_id}")
            
            # ✅ Also cache in VERTICAL_PARAM_CACHE for use in /chat route
            with CACHE_LOCK:
                VERTICAL_PARAM_CACHE[vertical_id] = df_params_local
                logging.info(f"✅ Cached parameter DF in VERTICAL_PARAM_CACHE for vertical {vertical_id}")
            
            # ✅ ADD THIS - Display param DF like device DF
            logging.info(f"\n{'='*80}")
            logging.info(f"📋 PARAMETERS DF for vertical {vertical_id}:")
            logging.info("\n" + df_params_local.to_string(index=False))
            logging.info(f"{'='*80}\n")

    except Exception as e:
        logging.error(f"❌ Failed to load parameters for vertical {vertical_id}: {e}")
        PARAM_CACHE[vertical_id] = pd.DataFrame(columns=["device_id", "device_name", "pin", "parameter"])

    logging.info(f"🔍 Loaded {len(df_devices_local)} devices and {len(PARAM_CACHE[vertical_id])} params into cache for vertical {vertical_id}")

    # Return success (single return)
    return jsonify({
        "success": True,
        "username": user_data.get('username'),
        "vertical_id": user_data.get('vertical_id'),
        "vertical_name": user_data.get('vertical_name')
    }), 200


# ==================== RUN ====================
if __name__ == '__main__':
    logging.info("STARTING FLASK SERVER")
    app.run(host='0.0.0.0', port=5000, debug=True)