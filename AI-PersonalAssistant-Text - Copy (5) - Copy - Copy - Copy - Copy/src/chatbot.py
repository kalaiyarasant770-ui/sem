

# chatbot.py

import json
import logging
import time
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, TypedDict
from openai import OpenAI
from langgraph.graph import StateGraph, END

from config import OPENAI_API_KEY, OPENAI_MODEL, SYSTEM_PROMPT, SCHEMA_CONTEXT
from database import get_tables_for_time_range, engine, text, validate_generated_sql
from utils import  _get_efficiency_note, generate_excel_report_in_memory
from chiller import calculate_device_metrics_with_llm
from dateutil.relativedelta import relativedelta

from typing import TypedDict, Dict, Any, List, Optional
from datetime import datetime,timedelta
import time
import base64
from memory import redis_client
from parameter_query import detect_list_parameters_query, check_exit_command, detect_time_only

from pydantic import BaseModel

from parameter_query import (
    detect_parameter_query, 
    handle_parameter_query,
    get_available_devices_message,
    check_exit_command
)

# Initialize OpenAI client
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# RAG system (will be set by main.py)
rag_system = None

logging.info("✅ Chatbot module loaded")

from typing import TypedDict, Dict, Any, List,Optional

class ChatbotState(BaseModel):
    user_message: str
    session_id: str
    conversation_history: list
    intent_data: dict
    response_data: dict
    final_response: str
    response_time: float
    start_time: datetime

    # LLM helper context
    parameter_list_for_llm: list = []
    pin_context_for_llm: str = ""
    device_list_for_llm: list = []   # list of devices (id/name/type) fed from main.py

    # === FOLLOW-BACK STATE (WITH DEFAULTS) ===
    awaiting_device_input: bool = False
    # Support both names because different parts used different keys in earlier code
    pending_parameter_query: Optional[str] = None
    pending_parameter_name: Optional[str] = None
    pending_time_range: Optional[Dict] = None
    pending_device_id: Optional[int] = None

# ==================== INTENT CLASSIFICATION ====================

def classify_query_intent(query: str, conversation_history: list = None, device_list: list = None) -> Dict[str, Any]:
    """Context-aware intent classification with STRICT ordering."""
    logging.info(f"\n{'='*80}")
    logging.info(f"🧠 INTENT CLASSIFICATION START")
    logging.info(f"{'='*80}")
    logging.info(f"Query: {query}")
    
    current_time = datetime.now()
    
    # ✅ STEP 1: Check exit commands (return immediately - no time needed)
    if check_exit_command(query):
        logging.info(f"   🚪 Exit command detected")
        return {
            'intent': 'exit',
            'confidence': 1.0,
            'original_query': query
        }
    
    # ✅ STEP 2: Check for "list parameters" query (return immediately - no time needed)
    list_check = detect_list_parameters_query(query)
    logging.info(f" 🔍 List parameters check: {list_check}")
    if list_check:
        logging.info(f"   📋 List parameters query detected")
        return {
            'intent': 'list_parameters',
            'confidence': 1.0,
            'original_query': query
        }
    
    # -----------------------------------------
    # STEP 3 — Try parameter query detection EARLY
    # -----------------------------------------
    param_detection = detect_parameter_query(query, device_list=device_list)

    if param_detection.get("is_parameter_query") and param_detection.get("confidence", 0) >= 0.60:
        logging.info(f"   ✅ EARLY DETECTION: Parameter Query (confidence: {param_detection['confidence']})")

        # ✅ Extract conversation context first
        last_time_start = None
        last_time_end = None
        last_intent = None  # ADD THIS
        
        if conversation_history and len(conversation_history) > 0:
            for msg in reversed(conversation_history):
                if msg.get('role') == 'assistant':
                    content = msg.get('content', '')
                    
                    # Detect previous intent
                    if 'Excel report generated' in content or '.xlsx' in content:
                        last_intent = 'report'
                    elif 'Cooling Efficiency' in content or 'Performance:' in content:
                        last_intent = 'metrics'
                    elif 'maintenance' in content.lower() or 'advice' in content.lower():
                        last_intent = 'advisory'
                    
                    import re
                    time_matches = re.findall(r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})', content)
                    if len(time_matches) >= 2:
                        last_time_start = time_matches[0]
                        last_time_end = time_matches[1]
                    
                    break
        
        # ✅ Check for follow-up phrases
        query_lower = query.lower()
        followup_indicators = ['what about', 'about', 'how about', 'same for', 'same time', 'also', 'and']
        is_followup = any(phrase in query_lower for phrase in followup_indicators)
        
        # ✅ CRITICAL FIX: ALWAYS extract time for parameter queries
        logging.info(f"\n--- Extracting Time Range for PARAMETER QUERY ---")
        
        previous_time_context = "None"
        if last_time_start and last_time_end:
            previous_time_context = f"{last_time_start} to {last_time_end}"
        
        time_prompt = f"""
    Extract time range from query OR use previous time from conversation history.

    Current Query: "{query}"
    Current Date: {current_time.strftime('%Y-%m-%d')}
    Previous Time Range: {previous_time_context}
    Is Follow-up Query: {is_followup}

    CRITICAL TIME PARSING RULES:
    1. "past 3 months" → Calculate EXACTLY 3 months back from today
    2. "june to june 10" → 2025-06-01T00:00:00 to 2025-06-10T23:59:59
    3. "jan 20 to 25" → 2025-01-20T00:00:00 to 2025-01-25T23:59:59
    4. "past 4 months" → Calculate 4 months back
    5. "last 3 months" → Same as "past 3 months"
    6. "for past 50 years" → Calculate 50 years back (USE EXACT USER INPUT)
    7. If follow-up with NO time words → USE previous time
    8. If year not mentioned → use CURRENT YEAR (2025)
    9. NEVER use future dates

    Return JSON (MUST include all fields):
    {{
        "start_time": "YYYY-MM-DDTHH:MM:SS" or null,
        "end_time": "YYYY-MM-DDTHH:MM:SS" or null,
        "source": "explicit" or "history" or "none",
        "reasoning": "<explain your calculation>"
    }}
    """
        
        start_time = None
        end_time = None
        time_source = 'none'
        
        try:
            time_response = openai_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": time_prompt}],
                response_format={"type": "json_object"},
                temperature=0,
                max_tokens=250
            )

            time_data = json.loads(time_response.choices[0].message.content)
            start_time = time_data.get('start_time')
            end_time = time_data.get('end_time')
            time_source = time_data.get('source', 'none')
            reasoning = time_data.get('reasoning', '')
            
            logging.info(f"   🕐 Time extraction raw: {start_time} to {end_time}")
            logging.info(f"   📝 Source: {time_source} - {reasoning}")

            # Normalize end time to end-of-day
            if start_time:
                try:
                    start_dt = datetime.fromisoformat(start_time)
                    start_time = start_dt.strftime('%Y-%m-%dT%H:%M:%S')
                except Exception:
                    logging.warning("   ⚠️ Start time parse failed; leaving as-is")
            if end_time:
                try:
                    end_dt = datetime.fromisoformat(end_time)
                    if end_dt.hour == 0 and end_dt.minute == 0 and end_dt.second == 0:
                        end_dt = end_dt.replace(hour=23, minute=59, second=59)
                    end_time = end_dt.strftime('%Y-%m-%dT%H:%M:%S')
                except Exception:
                    logging.warning("   ⚠️ End time parse failed; leaving as-is")
            
            logging.info(f"   ✅ Time extracted: {start_time} to {end_time}")
            
        except Exception as e:
            logging.error(f"   ❌ Time extraction failed: {e}")
        
        # if no device mentioned -> go to follow-back
        result = {
            'intent': 'parameter_query',
            'confidence': param_detection['confidence'],
            'parameter_name': param_detection.get('parameter_name'),
            'original_query': query,
            'time_info': {
                'start_time': start_time,
                'end_time': end_time,
                'source': time_source
            },
            'is_followup': False,
            'previous_intent': None,
        }
        
        logging.info(f"   🎯 EARLY PARAM QUERY RESULT: time={start_time} to {end_time}")
        return result
    
    # ✅ STEP 3: Extract conversation context
    last_intent = None
    last_time_start = None
    last_time_end = None
    
    if conversation_history and len(conversation_history) > 0:
        for msg in reversed(conversation_history):
            if msg.get('role') == 'assistant':
                content = msg.get('content', '')
                
                # Detect previous intent
                if 'Excel report generated' in content or '.xlsx' in content:
                    last_intent = 'report'
                    logging.info(f"   📋 PREVIOUS ACTION: REPORT")
                elif 'Cooling Efficiency' in content or 'Performance:' in content:
                    last_intent = 'metrics'
                    logging.info(f"   📊 PREVIOUS ACTION: METRICS")
                elif 'maintenance' in content.lower() or 'advice' in content.lower():
                    last_intent = 'advisory'
                    logging.info(f"   💡 PREVIOUS ACTION: ADVISORY")
                
                # Extract previous time
                import re
                time_matches = re.findall(r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})', content)
                if len(time_matches) >= 2:
                    last_time_start = time_matches[0]
                    last_time_end = time_matches[1]
                    logging.info(f"   🕐 PREVIOUS TIME: {last_time_start} to {last_time_end}")
                
                break
    
    # ✅ STEP 4: Check for follow-up phrases
    query_lower = query.lower()
    followup_indicators = ['what about', 'about', 'how about', 'same for', 'same time', 'also', 'and']
    is_followup = any(phrase in query_lower for phrase in followup_indicators)
    
    logging.info(f"   Follow-up query: {is_followup}")
    
    # ✅ STEP 5: Detect intent (SET variable, don't return yet)
    determined_intent = None
    
    # 5a. Check for EXPLICIT keywords first (highest priority)
    if any(kw in query_lower for kw in ['report', 'excel', 'download', 'xlsx', 'logs', 'loggs']):
        determined_intent = 'report'
        logging.info(f"   📊 Explicit REPORT keyword detected")
    
    elif any(kw in query_lower for kw in ['cop', 'eer', 'performance', 'efficiency', 'perform', 'result']):
        determined_intent = 'metrics'
        logging.info(f"   📈 Explicit METRICS keyword detected")
    
    elif any(kw in query_lower for kw in ['how to', 'improve', 'optimize', 'maintain', 'advice', 'tips', 'reduce', 'how can']):
        determined_intent = 'advisory'
        logging.info(f"   💡 Explicit ADVISORY keyword detected")

    # 5b. Check for greetings BEFORE parameter query (looser match)
    if not determined_intent:
        query_stripped = query_lower.strip()
        greetings = [
            'hello', 'hi', 'hey', 'good morning', 'good evening', 'good afternoon',
            'gm', 'gn', 'good bye', 'bye', 'good night', 'thanks', 'thank you', 'thanku'
            ]
        # allow greetings anywhere for short messages
        if any((g in query_lower) for g in greetings) and len(query.split()) <= 4:
            determined_intent = 'greeting'
            logging.info(f"   👋 GREETING detected")
    
    # 5d. Follow-up without explicit keywords → preserve last intent
    if not determined_intent and is_followup and last_intent:
        determined_intent = last_intent
        logging.info(f"   ✅ FOLLOW-UP: Preserving last intent ({last_intent})")
    
    # 5e. LLM fallback if still no intent
    if not determined_intent:
        logging.info(f"   ⚠️  No clear intent, using LLM fallback")
        
        context = ""
        if conversation_history:
            context = "\n\nLast 2 messages:\n"
            for msg in conversation_history[-4:]:
                role = msg.get('role', 'user')
                content = msg.get('content', '')[:150]
                context += f"{role}: {content}...\n"
        
        llm_prompt = f"""
Quick intent classification.

{context}

Current Query: "{query}"

Return JSON with ONLY:
{{
    "intent": "metrics" or "advisory" or "report" or "greeting",
    "confidence": <0.0 to 1.0>
}}

Rules:
- Performance/efficiency questions → metrics
- How-to/maintenance questions → advisory
- Report/excel requests → report
- Casual chat → greeting
"""
        
        try:
            response = openai_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": llm_prompt}],
                response_format={"type": "json_object"},
                temperature=0,
                max_tokens=100
            )
            llm_result = json.loads(response.choices[0].message.content)
            determined_intent = llm_result.get('intent', 'greeting')
            logging.info(f"   🤖 LLM fallback: {determined_intent} (confidence: {llm_result.get('confidence', 0):.2f})")
        except Exception as e:
            logging.error(f"   ❌ LLM failed: {e}, defaulting to greeting")
            determined_intent = 'greeting'
    
    # ✅ STEP 6: Extract time range (ONLY for intents that need it)
    start_time = None
    end_time = None
    time_source = 'none'
    
    if determined_intent in ['metrics', 'report','parameter_query']:
        logging.info(f"\n--- Extracting Time Range for {determined_intent.upper()} ---")
        
        previous_time_context = "None"
        if last_time_start and last_time_end:
            previous_time_context = f"{last_time_start} to {last_time_end}"
        

        time_prompt = f"""
Extract time range from query OR use previous time from conversation history.

Current Query: "{query}"
Current Date: {current_time.strftime('%Y-%m-%d')}
Previous Time Range: {previous_time_context}
Is Follow-up Query: {is_followup}

CRITICAL TIME PARSING RULES:
1. "past 3 months" → Calculate EXACTLY 3 months back from today
2. "june to june 10" → 2025-06-01T00:00:00 to 2025-06-10T23:59:59
3. "past 4 months" → Calculate 4 months back
4. "last 3 months" → Same as "past 3 months"
5. "for past 50 years" → Calculate 50 years back (USE EXACT USER INPUT)
6. If follow-up with NO time words → USE previous time
7. If year not mentioned → use CURRENT YEAR (2025)
8. If "same time" is mentioned → USE previous time
8. NEVER use future dates

MONTH CALCULATION (for "past X months"):
- Current: {current_time.strftime('%Y-%m-%d')}
- "past 1 month" → Start: {(current_time - timedelta(days=30)).strftime('%Y-%m-%d')}
- "past 3 months" → Start: {(current_time - timedelta(days=90)).strftime('%Y-%m-%d')}
- "past 6 months" → Start: {(current_time - timedelta(days=180)).strftime('%Y-%m-%d')}
- "past 12 months" → Start: {(current_time - timedelta(days=365)).strftime('%Y-%m-%d')}

EXAMPLES:
✅ Query: "3rd chiller's refriger level for past 3 months?", Today: 2025-11-18
   → {{
      "start_time": "2025-08-19T00:00:00",
      "end_time": "2025-11-18T23:59:59",
      "source": "explicit",
      "reasoning": "Past 3 months = 3 * 30 days back"
   }}

✅ Query: "past 6 months", Today: 2025-11-18
   → {{
      "start_time": "2025-05-19T00:00:00",
      "end_time": "2025-11-18T23:59:59",
      "source": "explicit"
   }}

❌ Query: "show me voltage", Previous: None, Follow-up: false
   → {{
      "start_time": null,
      "end_time": null,
      "source": "none"
   }}

Return JSON (MUST include all fields):
{{
    "start_time": "YYYY-MM-DDTHH:MM:SS" or null,
    "end_time": "YYYY-MM-DDTHH:MM:SS" or null,
    "source": "explicit" or "history" or "none",
    "reasoning": "<explain your calculation>"
}}
"""
        
        try:
            time_response = openai_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": time_prompt}],
                response_format={"type": "json_object"},
                temperature=0,
                max_tokens=250
            )

            time_data = json.loads(time_response.choices[0].message.content)
            start_time = time_data.get('start_time')
            end_time = time_data.get('end_time')
            time_source = time_data.get('source', 'none')
            reasoning = time_data.get('reasoning', '')
            
            logging.info(f"   🕐 Time extraction raw: {start_time} to {end_time}")
            logging.info(f"   📍 Source: {time_source} - {reasoning}")

            # --- NORMALIZE LLM TIMES (if present) ---
            # Ensure end_time is end-of-day (23:59:59) unless LLM returned an exact timestamp with time
            def normalize_iso_date(ts):
                try:
                    # If LLM gave date only or date+time, try to parse
                    dt = datetime.fromisoformat(ts)
                except Exception:
                    return None
                # If time is midnight 00:00:00 and user likely gave a date range, keep as-is
                # Otherwise if user gave a date (no time component), set end to 23:59:59
                if dt.hour == 0 and dt.minute == 0 and dt.second == 0:
                    # We'll treat this as date-only and set to end-of-day only for end_time
                    return dt
                return dt

            if start_time:
                try:
                    # convert to datetime to validate format
                    start_dt = datetime.fromisoformat(start_time)
                    start_time = start_dt.strftime('%Y-%m-%dT%H:%M:%S')
                except Exception:
                    logging.warning("   ⚠️ Start time parse failed; leaving as-is")
            if end_time:
                try:
                    end_dt = datetime.fromisoformat(end_time)
                    # If LLM gave date only (time 00:00:00) or a date-range, make end inclusive (23:59:59)
                    if end_dt.hour == 0 and end_dt.minute == 0 and end_dt.second == 0:
                        end_dt = end_dt.replace(hour=23, minute=59, second=59)
                    end_time = end_dt.strftime('%Y-%m-%dT%H:%M:%S')
                except Exception:
                    logging.warning("   ⚠️ End time parse failed; leaving as-is")


            # ---------- NEW: Intent-based DEFAULT time ranges ----------
            # If LLM failed to return times, set safe defaults per intent
            if not start_time or not end_time:
                now = datetime.now()
                # metrics -> last 2 days
                if determined_intent == 'metrics':
                    end_time = now.strftime('%Y-%m-%dT%H:%M:%S')
                    start_time = (now - timedelta(days=2)).strftime('%Y-%m-%dT%H:%M:%S')
                    time_source = 'default_metrics_2d'
                    logging.info(f"   🔧 Applied default METRICS time: {start_time} to {end_time}")
                # report -> last 1 month
                elif determined_intent == 'report':
                    end_time = now.strftime('%Y-%m-%dT%H:%M:%S')
                    start_time = (now - relativedelta(months=1)).strftime('%Y-%m-%dT%H:%M:%S')
                    time_source = 'default_report_1m'
                    logging.info(f"   🔧 Applied default REPORT time: {start_time} to {end_time}")
                # parameter_query -> last 2 months
                elif determined_intent == 'parameter_query':
                    end_time = now.strftime('%Y-%m-%dT%H:%M:%S')
                    start_time = (now - relativedelta(months=2)).strftime('%Y-%m-%dT%H:%M:%S')
                    time_source = 'default_param_2m'
                    logging.info(f"   🔧 Applied default PARAMETER time: {start_time} to {end_time}")
            
            # ✅ FALLBACK: If LLM returned null but we have previous time and it's a follow-up
            if (not start_time or not end_time) and is_followup and last_time_start and last_time_end:
                logging.warning(f"   ⚠️  LLM returned null, forcing previous time for follow-up")
                start_time = last_time_start
                end_time = last_time_end
                time_source = 'history'
            
        except Exception as e:
            logging.error(f"   ❌ Time extraction failed: {e}")
            
            # ✅ EMERGENCY FALLBACK: Use previous time if available and it's a follow-up
            if is_followup and last_time_start and last_time_end:
                logging.warning(f"   🔧 Using previous time as emergency fallback")
                start_time = last_time_start
                end_time = last_time_end
                time_source = 'history'
    
    # ✅ STEP 7: Build and return final result
    result = {
        'intent': determined_intent,
        'confidence': 0.95,
        'device_name': None,
        'time_info': {
            'start_time': start_time,
            'end_time': end_time,
            'source': time_source
        },
        'original_query': query,
        'is_followup': is_followup,
        'previous_intent': last_intent
    }
    
    # ✅ ADD: Include parameter_name if it's a parameter query
    if determined_intent == 'parameter_query' and param_detection:  # ← REUSE stored result
        result['parameter_name'] = param_detection.get('parameter_name')
        logging.info(f"   📌 Added parameter_name: {result['parameter_name']}")

    logging.info(f"\n✅ FINAL CLASSIFICATION:")
    logging.info(f"   Intent: {determined_intent}")
    logging.info(f"   Time: {start_time} to {end_time} (source: {time_source})")
    logging.info(f"   Follow-up: {is_followup}, Previous: {last_intent}")
    logging.info(f"{'='*80}\n")
    
    return result
# ==================== WORKFLOW NODES ====================

def classify_intent_node(state: ChatbotState) -> ChatbotState:
    """Node: Classify user intent. Handles follow-back device & time-only followups."""
    logging.info(f"📍 NODE: classify_intent")

    msg = (state.user_message or "").strip().lower()

    # ✅ STEP 1: Check if preserved data exists AND user message is a device name
    has_preserved_data = (
        state.intent_data and 
        state.intent_data.get('parameter_name') and 
        state.intent_data.get('time_info') and
        state.intent_data.get('time_info').get('start_time')
    )
    
    # ✅ NEW: Check if it's marked as device-only query from main.py
    is_device_only = state.intent_data.get('is_device_only', False) if state.intent_data else False
    
    # Check if user is asking something NEW (not just device name)
    user_msg_lower = (state.user_message or "").lower()
    
    # Keywords that indicate NEW query (not follow-back completion)
    new_query_keywords = [
        'report', 'excel', 'download', 'performance', 'metrics','cop', 'performs', 'efficiency', 'status',
        'what about', 'how about',
        'what are', 'list', 'show','pin', 'parameter',
        'hello', 'hi', 'thanks',
        'how to', 'improve', 'optimize','tips',
        'average', 'frequency', 'voltage', 'current', 'power'
    ]
    
    # ✅ CRITICAL FIX: If it's a device-only query, don't treat as new query
    is_new_query = (not is_device_only) and any(kw in user_msg_lower for kw in new_query_keywords) and len(user_msg_lower.split()) > 3
    
    if has_preserved_data and not is_new_query:
        logging.info("🔄 PRESERVED DATA DETECTED (device-only or follow-back)")
        
        device_name = state.intent_data.get('device_name')
        state.intent_data['device_name'] = device_name

        logging.info(f"   Device: {device_name}")
        logging.info(f"   Parameter: {state.intent_data.get('parameter_name')}")
        logging.info(f"   Time: {state.intent_data.get('time_info')}")
        
        # Force parameter_query intent
        state.intent_data['intent'] = 'parameter_query'
        state.awaiting_device_input = False
        
        logging.info(f"✅ FORCED INTENT: parameter_query (using preserved data)")
        return state
    
    # If preserved data exists but it's a NEW query, clear it
    if has_preserved_data and is_new_query:
        logging.info(f"🗑️ Clearing preserved data - detected new query type")
        state.intent_data.pop('parameter_name', None)
        state.intent_data.pop('time_info', None)
    
    # ✅ STEP 2: Check if we're in follow-back mode (old logic)
    if state.awaiting_device_input:
        logging.info("🔄 FOLLOW-BACK MODE DETECTED")
        logging.info(f"   Device already matched: {state.intent_data.get('device_name')}")
        logging.info(f"   Parameter already set: {state.intent_data.get('parameter_name')}")
        logging.info(f"   Time already set: {state.intent_data.get('time_info')}")
        
        # FORCE parameter_query intent - do NOT call classify_query_intent
        state.intent_data = state.intent_data or {}
        state.intent_data['intent'] = 'parameter_query'
        
        # Clear follow-back flag
        state.awaiting_device_input = False
        
        logging.info(f"✅ FORCED INTENT: parameter_query")
        return state

    # ✅ STEP 3: Normal classification (NOT in follow-back, no preserved data)
    logging.info(f"📍 NORMAL CLASSIFICATION MODE")
    
    intent_data = classify_query_intent(
        state.user_message,
        conversation_history=state.conversation_history or [],
        device_list=state.device_list_for_llm or []
    )

    detected_intent = intent_data.get("intent")
    logging.info(f"   Detected intent: {detected_intent}")

    # ✅ NEW: Check if intent changed from parameter_query
    last_intent = None
    for msg in reversed(state.conversation_history or []):
        if '[PARAM_CONTEXT]' in msg.get('content', ''):
            last_intent = 'parameter_query'
            break
        elif '[FOLLOW_BACK_STATE]' in msg.get('content', ''):
            last_intent = 'parameter_query'
            break

    # If intent changed → Clear ALL parameter context
    if last_intent == 'parameter_query' and detected_intent != 'parameter_query':
        logging.info(f"🗑️ Intent changed ({last_intent} → {detected_intent}) → Clearing ALL context")
        
        # Clear follow-back markers
        state.conversation_history = [
            msg for msg in state.conversation_history 
            if '[PARAM_CONTEXT]' not in msg.get('content', '') and 
            '[FOLLOW_BACK_STATE]' not in msg.get('content', '')
        ]
        
        # Clear state variables
        state.awaiting_device_input = False
        state.pending_parameter_query = None
        state.pending_time_range = None
        
        # Clear from Redis
        from memory import redis_client, add_message_to_history
        if redis_client and state.session_id:
            try:
                redis_client.delete(f"chat:{state.session_id}")
                for msg in state.conversation_history:
                    add_message_to_history(state.session_id, msg['role'], msg['content'])
                logging.info(f"   ✅ Context cleared from Redis")
            except Exception as e:
                logging.error(f"   ❌ Failed to clean Redis: {e}")

    # If intent is NOT parameter_query, clear any follow-back state
    elif detected_intent != "parameter_query":
        logging.info(f"🔄 Non-parameter intent ({detected_intent}) → clearing follow-back")
        state.awaiting_device_input = False
        state.pending_parameter_query = None
        state.pending_time_range = None

    # Preserve device info from existing state (do not overwrite)
    existing_intent = state.intent_data or {}
    if existing_intent:
        for key in ("device_id", "device_name", "device_type", "device_type_name", "cluster_id", "cluster_name", "slave_id"):
            if existing_intent.get(key) is not None:
                intent_data[key] = existing_intent.get(key)

    state.intent_data = intent_data

    logging.info(f"✅ CLASSIFICATION COMPLETE:")
    logging.info(f"   Intent: {intent_data.get('intent')}")
    logging.info(f"   Device: {intent_data.get('device_name')} (ID: {intent_data.get('device_id')})")
    logging.info(f"   Parameter: {intent_data.get('parameter_name')}")

    return state


def handle_metrics_node(state: ChatbotState) -> ChatbotState:
    """Node: Handle metrics queries."""
    logging.info(f"\n{'='*80}")
    logging.info(f"📍 NODE: handle_metrics")
    logging.info(f"{'='*80}")
    
    # query = state['user_message']
    query = state.user_message
    # intent_data = state['intent_data']
    intent_data = state.intent_data

    # GET DEVICE INFO
    device_id = intent_data.get('device_id')
    device_name = intent_data.get('device_name')
    device_type = intent_data.get('device_type')
    device_type_name = intent_data.get('device_type_name')
    
    logging.info(f"DEVICE: {device_name} (ID: {device_id})")
    logging.info(f"TYPE: {device_type_name} ({device_type})")
    
    if not device_id:
        logging.error("❌ No device_id in intent_data!")
        state.response_data = {
            "type": "error",
            "message": "Device not found."
        }
        return state
    
    time_info = intent_data.get('time_info', {})
   
    # ✅ REQUIRE time range for metrics
    if not time_info.get('start_time') or not time_info.get('end_time'):
        logging.error("❌ No time range provided for metrics query")
        state.response_data= {
            "type": "error",
            "message": "Please specify a time range (e.g., 'past 3 months', 'june to july')"
        }
        return state

    start_time = time_info.get('start_time', 'N/A')
    end_time = time_info.get('end_time', 'N/A')
    
    # ===================================================================
    # CHILLER METRICS (TYPE 1) - Full calculations
    # ===================================================================
    if device_type == 1:
        logging.info(f"✅ Device is a chiller - proceeding with metrics")
        logging.info(f"Query: {query}")
        logging.info(f"Device ID: {device_id}")
        logging.info(f"QUERY TYPE: Metrics")
        logging.info(f"DEVICE: {device_name} (ID: {device_id})")
        logging.info(f"DEVICE TYPE: {device_type_name} ({device_type})")
        logging.info(f"TIME RANGE: {start_time} to {end_time}")
        
        logging.info(f"Processing single device query...")
        sql_queries = generate_sql_query(query, device_id, time_info)
        
        if not sql_queries:
            state.response_data = {"type": "error", "message": "No valid tables"}
            return state
        
        logging.info(f"SQL QUERIES: {len(sql_queries)} generated")
        logging.info(f"Generated {len(sql_queries)} SQL queries")
        
        params = {'device_id': device_id}
        if 'start_time' in time_info:
            params['start'] = time_info['start_time']
            params['end'] = time_info['end_time']
            logging.info(f"Time params: {params['start']} to {params['end']}")
        
        dfs = []
        for idx, sql in enumerate(sql_queries, 1):
            logging.info(f"[{idx}/{len(sql_queries)}] Executing SQL...")
            
            validation = validate_generated_sql(sql, query)
            if not validation['is_valid']:
                logging.warning(f"   Validation failed: {validation['errors']}")
                continue
            
            try:
                with engine.connect() as connection:
                    result = connection.execute(text(sql), params).fetchall()
                if result:
                    df = pd.DataFrame(result)
                    dfs.append(df)
                    logging.info(f"   Retrieved {len(df)} rows")
                else:
                    logging.info(f"   No data")
            except Exception as e:
                if "doesn't exist" in str(e):
                    logging.warning(f"   Table missing (skipping)")
                else:
                    logging.error(f"   SQL error: {e}")
                continue
        
        if not dfs:
            state.response_data = {"type": "no_data", "message": "No data found in Required Time."}
            return state
        
        df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
        logging.info(f"Total data: {len(df)} rows")
        
        # Calculate metrics with device_name
        #metrics = calculate_device_metrics_with_llm(df, device_id, query, device_name)
        metrics = calculate_device_metrics_with_llm(
            df=df, 
            device_id=device_id, 
            query=query, 
            device_name=device_name,
            device_type=device_type,  # ✅ Pass device_type
            time_info=time_info        # ✅ Pass time_info
        )
        state.response_data = {"type": "metrics", "data": metrics}
    
    # ===================================================================
    # OTHER DEVICE TYPES - Placeholder (no calculations yet)
    # ===================================================================
    else:
        logging.warning(f"⚠️  Metrics not defined for device type {device_type} ({device_type_name})")
        logging.info(f"TIME RANGE: {start_time} to {end_time}")
        
        state.response_data = {
            "type": "metrics_placeholder",
            "data": {
                "device_id": device_id,
                "device_name": device_name,
                "device_type": device_type,
                "device_type_name": device_type_name,
                "start_time": start_time,
                "end_time": end_time,
                "message": f"Metrics calculation not yet implemented for {device_type_name} devices."
            }
        }
    logging.info(f"{'='*80}\n")
    return state

def handle_advisory_node(state: ChatbotState) -> ChatbotState:
    """Node: Handle advisory queries - NO DEVICE REQUIRED."""
    logging.info(f"\n{'='*80}")
    logging.info(f"📍 NODE: handle_advisory")
    logging.info(f"{'='*80}")
    
    # query = state['user_message']
    query = state.user_message
    logging.info(f"Query: {query}")
     
    # Search RAG system for relevant documentation
    rag_results = []
    if rag_system:
        logging.info("🔍 Searching RAG system...")
        rag_results = rag_system.search(query, top_k=3)
        logging.info(f"   Found {len(rag_results)} relevant chunks")
    else:
        logging.warning("⚠️  RAG system not available")
    
    context = ""
    if rag_results:
        context = "\n\nRelevant Documentation:\n"
        for i, result in enumerate(rag_results, 1):
            context += f"{i}. {result['text'][:200]}...\n"
    
    prompt = f"""
Provide brief maintenance advice in NUMBERED LIST format.

User Query: {query}
{context}

Instructions:
- Return numbered list: "1. ", "2. ", "3. "
- Maximum 5 points
- Be GENERAL maintenance advice (NOT device-specific)
- Focus on best practices, seasonal tips, operational optimization
- Do NOT mention specific device names or models
"""
    
    try:
        logging.debug("🤖 Generating advisory response with LLM...")
        response = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300
        )
        recommendations = response.choices[0].message.content.strip()
        logging.info(f"✅ Advisory response generated ({len(recommendations)} chars)")
    except Exception as e:
        logging.error(f"❌ LLM failed: {e}, using fallback")
        recommendations = "For optimal performance: 1) Regular maintenance checks, 2) Monitor operating conditions, 3) Follow preventive maintenance schedule."
    
    state.response_data = {
        "type": "advisory",
        "data": {
            "recommendations": recommendations
        }
    }
    
    logging.info(f"{'='*80}\n")
    return state

def handle_report_node(state: ChatbotState) -> ChatbotState:
    """Node: Handle Excel report generation."""
    logging.info(f"\n{'='*80}")
    logging.info(f"📊 NODE: handle_report")
    logging.info(f"{'='*80}")
    
    # intent_data = state['intent_data']
    intent_data = state.intent_data
    device_id = intent_data.get('device_id')
    device_name = intent_data.get('device_name')
    device_type = intent_data.get('device_type')
    cluster_id = intent_data.get('cluster_id')  
    time_info = intent_data.get('time_info', {})

    # # ✅ ADD THIS: Fallback for missing time range
    if not time_info.get('start_time') or not time_info.get('end_time'):
        from datetime import datetime, timedelta
        
        # ✅ REQUIRE time range for reports
        if not time_info.get('start_time') or not time_info.get('end_time'):
            logging.error("❌ No time range provided for report")
            state.response_data = {
                "type": "error",
                "message": "Please specify a time range for the report (e.g., 'june 1 to 10', 'past 2 months')"
            }
            return state
        end_time = datetime.now()
        start_time = end_time - timedelta(days=90)
        time_info = {
            'start_time': start_time.strftime('%Y-%m-%dT%H:%M:%S'),
            'end_time': end_time.strftime('%Y-%m-%dT%H:%M:%S')
        }
        logging.info(f"   Default time: {time_info['start_time']} to {time_info['end_time']}")
    
    logging.info(f"Generating Excel report for {device_name} (ID: {device_id}, Type: {device_type}, Cluster: {cluster_id})")
    logging.info(f"   Time range: {time_info.get('start_time')} to {time_info.get('end_time')}")
    logging.info(f"   Is follow-up: {intent_data.get('is_followup', False)}")
    
    # Generate Excel
    result = generate_excel_report_in_memory(
        device_id=device_id,
        time_info=time_info,
        # user_query=state['user_message'],
        user_query=state.user_message,
        device_name=device_name,
        device_type=device_type,
        cluster_id=cluster_id  # ← PASS cluster_id
    )
    
    if result['success']:
        state.response_data = {
            "type": "report",
            "data": result
        }
    else:
        state.response_data = {
            "type": "error",
            "message": "No Data for the device specific time."
        }
    
    logging.info(f"{'='*80}\n")
    logging.info(f"Generating Excel report for {device_name} (ID: {device_id}, Type: {device_type}, Cluster: {cluster_id})")
    logging.info(f"   Time range: {time_info.get('start_time')} to {time_info.get('end_time')}")
    logging.info(f"   Is follow-up: {intent_data.get('is_followup', False)}")
    return state

def handle_parameter_query_node(state: ChatbotState) -> ChatbotState:
    """Node: Handle parameter queries (supports follow-back for missing device/time)."""
    logging.info(f"\n{'='*80}")
    logging.info(f"📍 NODE: handle_parameter_query")
    logging.info(f"{'='*80}")

    state.intent_data = state.intent_data or {}
    intent_data = state.intent_data

    device_id = intent_data.get('device_id')
    device_name = intent_data.get('device_name')
    cluster_id = intent_data.get('cluster_id')
    parameter_name = intent_data.get('parameter_name')
    time_info = intent_data.get('time_info') or {}

    logging.info(f"Initial state:")
    logging.info(f"   Device: {device_name} (ID: {device_id})")
    logging.info(f"   Parameter: {parameter_name}")
    logging.info(f"   Time: {time_info.get('start_time')} to {time_info.get('end_time')}")

    # ==================== CHECK 1: All requirements met? ====================
    if device_id and parameter_name:
        logging.info(f"✅ All requirements present (device + parameter) → VALIDATE")
        
        # ✅ CRITICAL: Validate parameter exists for device BEFORE execution
        from parameter_query import resolve_parameter_to_pin
        
        param_info = resolve_parameter_to_pin(cluster_id, device_id, parameter_name)
        
        if not param_info:
            # Parameter doesn't exist for this device
            from parameter_query import get_available_parameters_for_device
            available_msg = get_available_parameters_for_device(cluster_id, device_id, device_name)
            
            logging.warning(f"⚠️ Parameter '{parameter_name}' not available for {device_name}")
            state.response_data = {"type": "error", "message": available_msg}
            logging.info(f"{'='*80}\n")
            return state
        
        if param_info.get('ambiguous'):
            # Parameter is ambiguous
            alternatives = param_info.get('alternatives', [])
            message = f"Multiple parameters match '{parameter_name}' for {device_name}. Please specify:\n\n"
            message += "\n".join(f"• {alt}" for alt in alternatives)
            
            logging.warning(f"⚠️ Ambiguous parameter match")
            state.response_data = {"type": "error", "message": message}
            logging.info(f"{'='*80}\n")
            return state
        
        # ✅ Parameter validated! Set default time if missing
        if not time_info or not time_info.get('start_time'):
            # Check for previous time in conversation
            has_time_in_history = False
            previous_time = None
            
            for msg in reversed(state.conversation_history):
                if '[PARAM_CONTEXT]' in msg.get('content', ''):
                    import re
                    time_start_match = re.search(r'time_start="([^"]*)"', msg['content'])
                    time_end_match = re.search(r'time_end="([^"]*)"', msg['content'])
                    
                    if time_start_match and time_end_match:
                        start = time_start_match.group(1)
                        end = time_end_match.group(1)
                        if start != 'None' and end != 'None':
                            previous_time = {
                                'start_time': start,
                                'end_time': end,
                                'source': 'history'
                            }
                            has_time_in_history = True
                            break
            
            if has_time_in_history and previous_time:
                time_info = previous_time
                logging.info(f"⏱️  Using previous time from context: {time_info}")
                state.intent_data['time_info'] = time_info
            else:
                # Use default: last 60 days
                end_time = datetime.now()
                start_time = end_time - timedelta(days=60)
                time_info = {
                    'start_time': start_time.strftime('%Y-%m-%dT%H:%M:%S'),
                    'end_time': end_time.strftime('%Y-%m-%dT%H:%M:%S')
                }
                logging.info(f"⏱️  Using default time: Last 60 days")
                state.intent_data['time_info'] = time_info
        
        # Execute parameter query
        result = handle_parameter_query(
            user_query=state.user_message,
            device_id=device_id,
            device_name=device_name,
            cluster_id=cluster_id,
            parameter_name=parameter_name,
            time_info=time_info
        )

        if result.get('success'):
            state.response_data = {"type": "parameter_query", "data": result}
            
            # Store FULL context for follow-ups
            from memory import add_message_to_history
            
            time_start = time_info.get('start_time', 'None') if time_info else 'None'
            time_end = time_info.get('end_time', 'None') if time_info else 'None'
            
            memory_marker = {
                'role': 'system',
                'content': f'[PARAM_CONTEXT] device_id="{device_id}" device_name="{device_name}" cluster_id="{cluster_id}" last_parameter="{parameter_name}" time_start="{time_start}" time_end="{time_end}"'
            }
            add_message_to_history(state.session_id, 'system', memory_marker['content'])
            logging.info(f"💾 Stored FULL context: device={device_name}, param={parameter_name}, time={time_start} to {time_end}")
        else:
            state.response_data = {"type": "error", "message": result.get('message', 'Unknown error')}

        logging.info(f"{'='*80}\n")
        return state

    # ==================== CHECK 2: Missing device? ====================
    if not device_id:
        logging.info(f"⚠️  Missing device → ACTIVATE FOLLOW-BACK")
        
        # Store pending state
        state.awaiting_device_input = True
        state.pending_parameter_query = parameter_name
        state.pending_time_range = time_info or {}

        # Build device list message
        try:
            df = pd.DataFrame(state.device_list_for_llm or [])
            if not df.empty and "device_name" in df.columns:
                device_names = df["device_name"].tolist()
                device_list_msg = "Please specify which device:\n\n" + "\n".join(f"• {n}" for n in device_names)
            else:
                device_list_msg = "Please specify which device you are asking about."
        except Exception as e:
            logging.error(f"Failed building device list: {e}")
            device_list_msg = "Please specify which device you are asking about."

        state.response_data = {"type": "ask_device", "message": device_list_msg}
        state.final_response = device_list_msg
        
        logging.info(f"{'='*80}\n")
        return state

    # ==================== CHECK 3: Missing parameter? ====================
    if not parameter_name:
        logging.info(f"⚠️  Missing parameter → ASK FOR PARAMETER")
        
        from parameter_query import get_available_parameters_for_device
        message = get_available_parameters_for_device(cluster_id, device_id, device_name)
        
        state.response_data = {"type": "error", "message": message}
        state.final_response = message
        
        logging.info(f"{'='*80}\n")
        return state

    logging.info(f"{'='*80}\n")
    return state

def handle_list_parameters_node(state: ChatbotState) -> ChatbotState:
    """Node: Handle 'what parameters are available?' queries."""
    logging.info(f"\n{'='*80}")
    logging.info(f"📍 NODE: handle_list_parameters")
    logging.info(f"{'='*80}")
    
    intent_data = state.intent_data
    device_id = intent_data.get('device_id')
    device_name = intent_data.get('device_name')
    cluster_id = intent_data.get('cluster_id')
    
    if not device_id or not cluster_id:
        logging.error("❌ Missing device_id or cluster_id")
        state.response_data = {
            "type": "error",
            "message": "Please specify which device you want to see parameters for."
        }
        return state
    
    from parameter_query import get_available_parameters_for_device
    
    message = get_available_parameters_for_device(cluster_id, device_id, device_name)
    
    state.response_data = {
        "type": "list_parameters",
        "data": {"message": message}
    }
    
    logging.info(f"{'='*80}\n")
    return state


def handle_greeting_node(state: ChatbotState) -> ChatbotState:
    """Node: Handle greeting/casual queries."""
    logging.info(f"\n{'='*80}")
    logging.info(f"📍 NODE: handle_greeting")
    logging.info(f"{'='*80}")

    query = state.user_message  # ✅ FIXED

    logging.info(f"Query: {query}")

    prompt = f"""
    You are a helpful IoT Devices monitoring assistant. Respond naturally.
    User Query: {query}
    Keep response under 30 words, friendly but professional.
    """

    try:
        logging.debug("🤖 Generating greeting response...")
        response = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        message = response.choices[0].message.content.strip()
        logging.info("✅ Greeting response generated")
    except Exception as e:
        logging.error(f"❌ LLM failed: {e}, using fallback")
        message = "Hello! I'm here to help with DevsBot Assistant."

    state.response_data = {"type": "greeting", "data": {"message": message}}
    logging.info(f"{'='*80}\n")
    return state

def generate_response_node(state: ChatbotState) -> ChatbotState:
    """Node: Generate final response."""
    logging.info(f"📍 NODE: generate_response")

    response_data = state.response_data
    intent_data = state.intent_data

    if response_data.get("type") == "error":
        state.final_response = response_data["message"]

    elif response_data.get("type") == "no_data":
        state.final_response = response_data["message"]

    elif response_data.get("type") == "greeting":
        state.final_response = response_data["data"]["message"]

    elif response_data.get("type") == "list_parameters":
        state.final_response = response_data["data"]["message"]

    elif response_data.get("type") == "List_parameters":
        state.final_response = response_data['data']['message']            

    elif response_data.get("type") == "metrics":
        # Properly format metrics for frontend and include machine-readable ISO timestamps
        metrics = response_data["data"]

        # Safety checks
        if not metrics or (isinstance(metrics, dict) and metrics.get('error')):
            device_name = state.intent_data.get('device_name', 'Device')
            state.final_response = f"No valid data available for {device_name} in the specified time range. Please check the device or try a different time period."
            return state

        required_keys = ['device_id', 'cop', 'eer', 'cooling_capacity_tr', 'average_power_kw', 'energy_cost_inr']
        missing_keys = [k for k in required_keys if k not in metrics]
        if missing_keys:
            device_name = state.intent_data.get('device_name', 'Device')
            logging.error(f"❌ Missing metrics keys: {missing_keys}")
            state.final_response = f"Failed to calculate metrics for {device_name}. Incomplete data available."
            return state

        device_id = metrics.get('device_id')
        device_name = state.intent_data.get('device_name', 'Unknown Device')
        if not device_name or device_name == 'Unknown Device':
            device_name = f"Device {device_id}"

        # Use time_info if available, else use metric-provided times (if any)
        start_iso = intent_data.get('time_info', {}).get('start_time') or metrics.get('start_time') or ''
        end_iso = intent_data.get('time_info', {}).get('end_time') or metrics.get('end_time') or ''

        # also prepare readable times
        readable_start = start_iso
        readable_end = end_iso
        try:
            if start_iso:
                readable_start = datetime.fromisoformat(start_iso).strftime('%b %d, %Y')
            if end_iso:
                readable_end = datetime.fromisoformat(end_iso).strftime('%b %d, %Y')
        except Exception:
            pass

        cop_value = f"{metrics.get('cop', 0):.3f}" if metrics.get('cop') is not None else 'N/A'
        eer_value = f"{metrics.get('eer', 0):.3f}" if metrics.get('eer') is not None else 'N/A'
        cooling_capacity = f"{metrics.get('cooling_capacity_tr', 0):.3f}" if metrics.get('cooling_capacity_tr') is not None else 'N/A'
        avg_power = f"{metrics.get('average_power_kw', 0):.3f}" if metrics.get('average_power_kw') is not None else 'N/A'
        energy_consumed = f"{metrics.get('energy_consumed_kwh', 0):.3f}" if metrics.get('energy_consumed_kwh') is not None else 'N/A'
        energy_cost = f"₹{metrics.get('energy_cost_inr', 0):.2f}" if metrics.get('energy_cost_inr') is not None else 'N/A'
        run_time = f"{metrics.get('run_time_hours', 0):.2f}" if metrics.get('run_time_hours') is not None else 'N/A'
        efficiency_note = metrics.get('efficiency_note', '')

        state.final_response = f"""[METRICS_START]
{device_name} Performance:

Start Time: {readable_start}
End Time: {readable_end}

Cooling Efficiency (COP): {cop_value}
Energy Efficiency (EER): {eer_value}
Cooling Load: {cooling_capacity} TR
Power Taken (avg): {avg_power} kW
Power Consumed: {energy_consumed} kWh
Power Cost: {energy_cost}
Run Time: {run_time} hrs

Note: {efficiency_note}
[METRICS_END]"""

    elif response_data.get("type") == "metrics_placeholder":
        data = response_data["data"]
        state.final_response = f" {data['message']}"

    elif response_data.get("type") == "advisory":
        recommendations = response_data["data"]["recommendations"]
        state.final_response = f"[ADVISORY_START]\n{recommendations}\n[ADVISORY_END]"

    elif response_data.get("type") == "parameter_query":
        data = response_data["data"]
        stats = data.get('statistics', {})

        param_name = data.get('parameter')
        device_name = data.get('device_name')
        pin = data.get('pin')

        start_time = stats.get('start_time', 'N/A')
        end_time = stats.get('end_time', 'N/A')

        if start_time != 'N/A':
            try:
                start_dt = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
                end_dt = datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S')
                start_time = start_dt.strftime('%b %d, %Y')
                end_time = end_dt.strftime('%b %d, %Y')
            except Exception as e:
                logging.warning(f"Time format conversion failed: {e}")

        state.final_response = f"""
{device_name} - {param_name} ({pin}):

Time Range: {start_time} to {end_time}

Average: {stats.get('average', 0):.2f}
Minimum: {stats.get('minimum', 0):.2f}
Maximum: {stats.get('maximum', 0):.2f}
Median: {stats.get('median', 0):.2f}

Data Points: {stats.get('data_points', 0)}"""

    elif response_data.get("type") == "report":
        report_data = response_data["data"]

        if not report_data.get('success'):
            state.final_response = "Failed to generate Excel report. Please try again."
        else:
            import base64
            from memory import redis_client

            report_id = f"report_{state.session_id}_{int(time.time())}"
            excel_bytes = report_data['excel_bytes']

            if redis_client:
                redis_client.setex(
                    f"excel:{report_id}",
                    300,
                    base64.b64encode(excel_bytes).decode('utf-8')
                )
                logging.info(f"📦 Stored Excel report: {report_id}")

            device_name = intent_data.get('device_name', 'Device')
            row_count = report_data['row_count']
            file_size_kb = len(excel_bytes) / 1024

            state.final_response = f"✅ Excel report generated successfully!\n\nDevice: {device_name}\nRows: {row_count}\nSize: {file_size_kb:.2f} KB"

            state.response_data['download_ready'] = True
            state.response_data['report_id'] = report_id
            state.response_data['download_filename'] = f"{device_name.replace(' ', '_')}_report.xlsx"
            state.response_data['rows'] = row_count
            state.response_data['file_size'] = f"{file_size_kb:.2f} KB"

    else:
        state.final_response = "I couldn't process your request."

    state.response_time = (datetime.now() - state.start_time).total_seconds()
    logging.info(f"✅ Final response generated in {state.response_time:.2f} seconds")
    return state

# ==================== WORKFLOW SETUP ====================

def route_intent(state: ChatbotState) -> str:
    """Route to appropriate handler based on intent."""

    # FIX: use attribute access for pydantic model
    intent = state.intent_data.get('intent', 'metrics')

    logging.info(f"🔀 Routing to handler: {intent}")
    
    if intent == 'exit':
        return 'handle_greeting'
    elif intent == 'list_parameters':
        return 'handle_list_parameters'
    elif intent == 'report':
        return 'handle_report'
    elif intent == 'metrics':
        return 'handle_metrics'
    elif intent == 'advisory':
        return 'handle_advisory'
    elif intent == 'parameter_query':
        return 'handle_parameter_query'
    else:
        return 'handle_greeting'

# ==================== UPDATE WORKFLOW GRAPH ====================

logging.info("🔄 Building workflow graph...")
workflow = StateGraph(ChatbotState)

# Add ALL nodes first
workflow.add_node("classify_intent", classify_intent_node)
workflow.add_node("handle_metrics", handle_metrics_node)
workflow.add_node("handle_advisory", handle_advisory_node)
workflow.add_node("handle_report", handle_report_node)
workflow.add_node("handle_parameter_query", handle_parameter_query_node)
workflow.add_node("handle_list_parameters", handle_list_parameters_node)  # ← ADD THIS
workflow.add_node("handle_greeting", handle_greeting_node)
workflow.add_node("generate_response", generate_response_node)

workflow.set_entry_point("classify_intent")

# Add conditional routing
workflow.add_conditional_edges(
    "classify_intent",
    route_intent,
    {
        "handle_report": "handle_report",
        "handle_metrics": "handle_metrics",
        "handle_advisory": "handle_advisory",
        "handle_list_parameters": "handle_list_parameters",  # ← ADD THIS
        "handle_parameter_query": "handle_parameter_query",
        "handle_greeting": "handle_greeting"
    }
)

# Add edges to generate_response
workflow.add_edge("handle_metrics", "generate_response")
workflow.add_edge("handle_advisory", "generate_response")
workflow.add_edge("handle_report", "generate_response")
workflow.add_edge("handle_parameter_query", "generate_response")
workflow.add_edge("handle_list_parameters", "generate_response")  # ← ADD THIS
workflow.add_edge("handle_greeting", "generate_response")
workflow.add_edge("generate_response", END)

chatbot_graph = workflow.compile()
logging.info("✅ Workflow graph compiled successfully")

# Helper function for SQL generation
def generate_sql_query(query: str, device_id: int, time_info: Dict[str, Any]) -> list:
    """Generate SQL queries for given device and time range."""
    tables = get_tables_for_time_range(time_info)
    if not tables:
        return []
    
    sql_queries = []
    for table in tables:
        json_column = 'device_value' if table == "device_data" else 'raw_value'
        time_column = 'datetime' if table == "device_data" else 'updatedtime'
        
        sql = f"""
            SELECT 
                device_id,
                {time_column},
                JSON_UNQUOTE({json_column}->'$.V18') AS power,
                JSON_UNQUOTE({json_column}->'$.V19') AS temp_in,
                JSON_UNQUOTE({json_column}->'$.V20') AS temp_out
            FROM {table}
            WHERE 
                device_id = :device_id
                AND {time_column} BETWEEN :start AND :end
                AND CAST(JSON_UNQUOTE({json_column}->'$.V18') AS DECIMAL(10,2)) > 5
                AND ABS(
                        (CAST(JSON_UNQUOTE({json_column}->'$.V20') AS DECIMAL(10,2)) 
                        - CAST(JSON_UNQUOTE({json_column}->'$.V19') AS DECIMAL(10,2)))
                    ) / 1.8 > 0.7
            """
        sql_queries.append(sql)
    
    return sql_queries
