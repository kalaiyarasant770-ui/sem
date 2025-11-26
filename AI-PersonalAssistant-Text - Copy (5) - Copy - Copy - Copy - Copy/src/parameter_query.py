

# src/parameter_query.py


"""
NEW FEATURE: Parameter Value Extraction
Handles queries like "What is the chiller 1 supply temperature?"
"""
import re
import json
import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple
from sqlalchemy import text
from openai import OpenAI

from config import OPENAI_API_KEY, OPENAI_MODEL
from database import engine, get_tables_for_time_range
from dateutil.relativedelta import relativedelta
# Initialize OpenAI client
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

def is_device_mentioned_in_query(query: str, device_list: Optional[list]) -> Optional[str]:
    """
    Deterministic check: if the user query contains a device name from device_list, return it.
    device_list can be list of dicts with 'device_name' or list of strings.
    """
    if not device_list:
        return None
    q = query.lower()
    for d in device_list:
        name = d.get('device_name') if isinstance(d, dict) else str(d)
        if not name:
            continue
        if name.lower() in q:
            return name
    return None

# ==================== PARAMETER DETECTION ====================

def detect_parameter_query(query: str, device_list=None, parameter_list=None, pin_context=None):
    """
    Detect if query is asking for a SINGLE PARAMETER VALUE.
    Uses deterministic keywords first, then LLM fallback.
    """
    logging.info(f"\n{'='*80}")
    logging.info(f"🔍 PARAMETER QUERY DETECTION")
    logging.info(f"{'='*80}")
    logging.info(f"Query: {query}")

    if not openai_client:
        logging.warning("⚠️  OpenAI client not available")
        return {'is_parameter_query': False, 'parameter_name': None, 'confidence': 0.0}
    
    # ✅ STEP 1: DETERMINISTIC CHECK - Keywords that indicate parameter query
    query_lower = query.lower()

    # ✅ NEW: Check if query is JUST a device name (no parameter keywords)
    if device_list:
        device_names_lower = [
            d.get('device_name').lower() if isinstance(d, dict) else str(d).lower() 
            for d in device_list
        ]
        
        # If query exactly matches a device name (no extra words)
        # Allow for simple ordinals like "2nd"
        query_clean = query_lower.replace("1st", "1").replace("2nd", "2").replace("3rd", "3")
        query_words = set(query_clean.split())
        for dev_name in device_names_lower:
            dev_words = set(dev_name.split())
            
            # If query words are the same as or a subset of device words
            if query_words == dev_words or query_words.issubset(dev_words):
                logging.info(f"   ⚠️ Query '{query}' is just a device name → NOT parameter query")
                return {'is_parameter_query': False, 'parameter_name': None, 'confidence': 1.0}
    
    # Parameter keywords (sensors/readings)
    param_keywords = [
    'frequency', 'voltage', 'current', 'power', 'energy', 'kwh', 'temperature', 
    'pressure', 'flow', 'level', 'speed', 'rpm', 'psi', 'amp', 'volt', 'hz',
    'supply', 'return', 'suction', 'discharge', 'readings', 'reading',
    'phase', 'ryb', 'line', 'neutral', 'factor', 'efficiency',
    # NOTE: 'cop' and 'eer' removed - these are METRICS, not parameters
    'capacity', 'load', 'consumption', 'usage'
  ]
    # Add separate check for metrics that should NOT be parameter queries
    metrics_keywords = ['cop', 'eer', 'performance', 'efficiency metric']

    # Reject keywords (metrics/reports/other)
    reject_keywords = [
        'report', 'excel', 'download', 'csv', 'logs', 'performance', 'compare', 
        'comparison', 'metrics', 'analysis', 'trend', 'how to', 'improve', 
        'optimize', 'maintain', 'advice', 'tips', 'list parameters', 'list pins',
        'show parameters', 'show pins', 'what parameters', 'which parameters'
    ]
    
    # Check reject keywords first
    for reject_kw in reject_keywords:
        if reject_kw in query_lower:
            logging.info(f"   ⚠️  DETERMINISTIC: Found reject keyword '{reject_kw}' → NOT parameter query")
            return {'is_parameter_query': False, 'parameter_name': None, 'confidence': 0.0}

    # âœ… NEW: Check metrics keywords (cop/eer should trigger metrics, not parameter query)
    if any(kw in query_lower for kw in metrics_keywords):
        logging.info(f"   ⚠️  DETERMINISTIC: Found metrics keyword → NOT parameter query")
        return {'is_parameter_query': False, 'parameter_name': None, 'confidence': 0.0}
        
    # Check for parameter keywords
    found_param_keywords = []
    for param_kw in param_keywords:
        if param_kw in query_lower:
            found_param_keywords.append(param_kw)
    
    if found_param_keywords:
        logging.info(f"   ✅ DETERMINISTIC: Found param keywords {found_param_keywords}")
        
        # ✅ FIX: Extract parameter WITH context words (average, ryb, phase, line, etc.)
        # Find the keyword position and grab surrounding words
        query_words = query_lower.split()
        
        # Find all words around the parameter keyword
        param_phrases = []
        for kw in found_param_keywords:
            for i, word in enumerate(query_words):
                if kw in word:
                    # Get 1-2 words before and the keyword itself
                    start = max(0, i - 2)
                    end = i + 1
                    phrase_words = query_words[start:end]
                    
                    # Filter out common words
                    skip_words = {'for', 'the', 'of', 'in', 'on', 'at', 'to', 'a', 'an', 'and'}
                    phrase_words = [w for w in phrase_words if w not in skip_words]
                    
                    if phrase_words:
                        param_phrases.append(' '.join(phrase_words))
        
        # Use the longest/most specific phrase found
        if param_phrases:
            parameter_name = max(param_phrases, key=len).title()
        else:
            parameter_name = found_param_keywords[0].title()
        
        logging.info(f"   Extracted parameter: '{parameter_name}'")
        logging.info(f"{'='*80}\n")
        return {
            'is_parameter_query': True,
            'parameter_name': parameter_name,
            'confidence': 0.95
        }

    # ✅ STEP 2: LLM FALLBACK (for edge cases)
    logging.info(f"   LLM FALLBACK: No deterministic match, using LLM")
    
    device_context = ""
    if device_list:
        names = [d.get('device_name') if isinstance(d, dict) else str(d) for d in device_list]
        device_context = "Available devices:\n" + "\n".join(f"- {n}" for n in names[:200]) + "\n\n"

    prompt = f"""
Classify: Is this asking for a SINGLE SENSOR VALUE or something else?

{device_context}

Query: "{query}"

Return JSON:
{{
  "is_parameter_query": true or false,
  "parameter_name": "<sensor name>" or null,
  "confidence": <0.0 to 1.0>
}}
"""

    try:
        response = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=100
        )
        result = json.loads(response.choices[0].message.content)
        is_param_query = result.get('is_parameter_query', False)
        param_name = result.get('parameter_name')
        confidence = result.get('confidence', 0.0)

        logging.info(f"✅ LLM Result:")
        logging.info(f"   Is parameter query: {is_param_query}")
        logging.info(f"   Parameter: {param_name}")
        logging.info(f"   Confidence: {confidence:.2f}")
        logging.info(f"{'='*80}\n")

        return {
            'is_parameter_query': is_param_query,
            'parameter_name': param_name,
            'confidence': confidence
        }

    except Exception as e:
        logging.error(f"❌ LLM Detection failed: {e}")
        logging.info(f"{'='*80}\n")
        return {'is_parameter_query': False, 'parameter_name': None, 'confidence': 0.0}
    
# ==================== ADD THIS TO parameter_query.py ====================
# Add this function after detect_parameter_query() function

def detect_time_only(query: str) -> Optional[dict]:
    """
    Detect if user message contains ONLY a time range
    (e.g. 'past 10 days', 'last week', 'today', 'yesterday')
    and NO parameter name, NO device name.

    Returns:
        {"start_time": "...", "end_time": "...", "source": "..."} 
        or None
    """
    import re
    from datetime import datetime, timedelta
    from dateutil.relativedelta import relativedelta

    q = query.lower().strip()

    # Basic patterns for time-only queries
    patterns = {
        r'past (\d+) days?': lambda d: (datetime.now() - timedelta(days=int(d)), datetime.now()),
        r'last (\d+) days?': lambda d: (datetime.now() - timedelta(days=int(d)), datetime.now()),
        r'past (\d+) months?': lambda m: (datetime.now() - relativedelta(months=int(m)), datetime.now()),
        r'last (\d+) months?': lambda m: (datetime.now() - relativedelta(months=int(m)), datetime.now()),
        r'past (\d+) weeks?': lambda w: (datetime.now() - timedelta(weeks=int(w)), datetime.now()),
        r'last (\d+) weeks?': lambda w: (datetime.now() - timedelta(weeks=int(w)), datetime.now()),
        r'today': lambda _=None: (datetime.now().replace(hour=0, minute=0, second=0), datetime.now()),
        r'yesterday': lambda _=None: (
            (datetime.now() - timedelta(days=1)).replace(hour=0, minute=0, second=0),
            (datetime.now() - timedelta(days=1)).replace(hour=23, minute=59, second=59)
        ),
    }

    for pattern, handler in patterns.items():
        m = re.search(pattern, q)
        if m:
            arg = m.group(1) if len(m.groups()) else None
            start, end = handler(arg)
            return {
                "start_time": start.strftime("%Y-%m-%dT%H:%M:%S"),
                "end_time": end.strftime("%Y-%m-%dT%H:%M:%S"),
                "source": "time_only_followup"
            }

    return None
    
# ==================== PARAMETER RESOLUTION ====================

def resolve_parameter_to_pin(cluster_id: int, device_id: int, parameter_name: str) -> Optional[Dict[str, Any]]:
    """
    Find pin number for a parameter name from datastream table.
    """
    logging.info(f"\n{'='*80}")
    logging.info(f"📌 PARAMETER → PIN RESOLUTION")
    logging.info(f"{'='*80}")
    logging.info(f"Cluster ID: {cluster_id}")
    logging.info(f"Device ID: {device_id}")
    logging.info(f"Parameter: {parameter_name}")
    
    sql = """
    SELECT pin, name, address
    FROM datastream
    WHERE cluster_Id = :cluster_id AND isdelete = 0
    ORDER BY pin
    """
    
    try:
        with engine.connect() as conn:
            result = conn.execute(text(sql), {"cluster_id": cluster_id}).fetchall()
        
        if not result:
            logging.warning(f"⚠️  No datastream found for cluster_id={cluster_id}")
            logging.info(f"{'='*80}\n")
            return None
        
        logging.info(f"Found {len(result)} datastream entries")
        
        # Collect all possible parameters for this device
        all_params = []
        
        for row in result:
            # ✅ FIX: Handle variable row length
            try:
                pin = row[0]  # Column 0: pin
                # name = row[1]  # Column 1: name (optional)
                address_json = row[2] if len(row) > 2 else None  # Column 2: address
            except IndexError:
                logging.warning(f"⚠️  Row has unexpected structure: {row}")
                continue
            
            if not address_json:
                continue
            
            try:
                address_array = json.loads(address_json)
                
                if not isinstance(address_array, list):
                    continue
                
                for address_obj in address_array:
                    if not isinstance(address_obj, dict):
                        continue
                    
                    obj_pin = address_obj.get('pin')
                    if obj_pin != pin:
                        continue
                    
                    # Check all address-X keys
                    address_keys = sorted([k for k in address_obj.keys() if k.startswith('address-')])
                    
                    for addr_key in address_keys:
                        addr_data = address_obj[addr_key]
                        
                        if not isinstance(addr_data, dict):
                            continue
                        
                        devices_list = addr_data.get('devices', [])
                        params_name = addr_data.get('params')
                        
                        # Check if this device is in the devices list
                        if isinstance(devices_list, list) and device_id in devices_list and params_name:
                            all_params.append({
                                'pin': pin,
                                'params': params_name,
                                'address_key': addr_key
                            })
                            logging.info(f"   Found: {pin} → {params_name} ({addr_key})")
            
            except json.JSONDecodeError:
                logging.warning(f"   ⚠️  Failed to parse JSON for pin {pin}")
                continue
        
        if not all_params:
            logging.warning(f"⚠️  No parameters found for device_id={device_id}")
            logging.info(f"{'='*80}\n")
            return None
        
        logging.info(f"\n📋 Available parameters for device {device_id}: {len(all_params)}")
        for p in all_params:
            logging.info(f"   - {p['params']} ({p['pin']})")
        
        # Use LLM to fuzzy match parameter name
        matched_param = _fuzzy_match_parameter(parameter_name, all_params)
        
        if matched_param:
            logging.info(f"\n✅ MATCHED:")
            logging.info(f"   Query parameter: '{parameter_name}'")
            logging.info(f"   Matched to: '{matched_param['params']}' ({matched_param['pin']})")
            logging.info(f"{'='*80}\n")
            return matched_param
        
        logging.warning(f"⚠️  No match found for '{parameter_name}'")
        logging.info(f"{'='*80}\n")
        return None
        
    except Exception as e:
        logging.error(f"❌ Failed to resolve parameter: {e}")
        logging.info(f"{'='*80}\n")
        return None
    
    
def _fuzzy_match_parameter(query_param: str, available_params: list) -> Optional[Dict[str, Any]]:
    """
    Try deterministic matching first (exact or token/subset matches, prefer 'today'/'ryb' tokens),
    then fallback to LLM fuzzy matching if needed.

    available_params: [{'pin': 'V20', 'params': 'Supply Temperature', ...}, ...]
    """
    if not available_params:
        return None

    # Normalize helpers
    def normalize(s: str) -> str:
        return re.sub(r'[^a-z0-9\s]', '', (s or '').lower()).strip()

    query_norm = normalize(query_param)
    query_tokens = set(query_norm.split())

    # Build list of candidate names (normalized) mapping to original param dict
    candidates = []
    for p in available_params:
        name = p.get('params') or ''
        name_norm = normalize(name)
        candidates.append((p, name, name_norm, set(name_norm.split())))

    # 1) Exact normalized match (best)
    for p, name, name_norm, toks in candidates:
        if query_norm == name_norm:
            return p

    # 2) Substring match (query contained in param name)
    for p, name, name_norm, toks in candidates:
        if query_norm in name_norm and len(query_norm) >= 2:
            return p

    # 3) Token containment: prefer parameters that contain all meaningful query tokens
    #    (Ignore tiny tokens like 'the', 'in', 'for')
    small_stop = {'the', 'in', 'for', 'and', 'of', 'a', 'an', 'to', 'today'}
    query_tokens_filtered = {t for t in query_tokens if t and t not in small_stop}

    if query_tokens_filtered:
        best = None
        best_score = 0.0
        for p, name, name_norm, toks in candidates:
            if not toks:
                continue
            
            # Calculate base match score
            match_count = len(query_tokens_filtered & toks)
            score = match_count / max(len(query_tokens_filtered), 1)
            
            # ✅ CRITICAL: Apply preference rules for ambiguous terms
            # If query is just "power" and name contains "average power" → BOOST
            if 'power' in query_tokens_filtered and len(query_tokens_filtered) == 1:
                if 'average' in toks:
                    score += 0.5  # Strong boost for "Average Power"
                elif 'ryb' in toks or 'phase' in toks:
                    score -= 0.3  # Penalty for "RYB Phase Power"
            
            # If query is just "voltage" and name contains "line voltage" → BOOST
            if 'voltage' in query_tokens_filtered and len(query_tokens_filtered) == 1:
                if 'line' in toks:
                    score += 0.5
                elif 'ryb' in toks or 'phase' in toks:
                    score -= 0.3
            
            # If query is just "current" → prefer RYB Phase Current (no change needed)
            
            # Original boost logic for specific cases
            boost = 0
            if 'today' in query_tokens and 'today' in toks:
                boost += 1
            if 'ryb' in query_tokens and 'ryb' in toks:
                boost += 1
            score += 0.1 * boost
            
            # ✅ Prefer SHORTER parameter names when scores are equal (simpler is better)
            if score == best_score and best is not None:
                current_len = len(name.split())
                best_len = len(best['params'].split())
                if current_len < best_len:
                    best = p
                    best_score = score
            elif score > best_score:
                best_score = score
                best = p

        # Lower threshold to 0.4 since we have preference rules now
        if best and best_score >= 0.4:
            logging.info(f"   ✅ Deterministic match: '{best['params']}' (score: {best_score:.2f})")
            return best

    # 4) As fallback, keep your existing LLM-based fuzzy match (if openai_client exists)
    if openai_client:
        try:
            params_list = [p['params'] for p in available_params]

            prompt = f"""
Match the user's parameter name to the closest available parameter.

User asked for: "{query_param}"

Available parameters:
{json.dumps(params_list, indent=2)}

CRITICAL MATCHING RULES:
1. "power" alone → prefer "Average Power" over "RYB Phase Power" or "Power Factor"
2. "voltage" alone → prefer "Line Voltage" over "RYB Phase Voltage"
3. "current" alone → prefer "RYB Phase Current"
4. If user says "ryb power" or "phase power" → then use "RYB Phase Power"
5. Generic terms prefer SIMPLER/SHORTER parameter names
6. Exact word match has priority over partial
7. "power" alone → prefer "Average Power" over "RYB Phase Power" or "Power Factor"
8. "voltage" alone → prefer "Line Voltage" over "RYB Phase Voltage"

EXAMPLES:
- "power" → "Average Power" (not "RYB Phase Power")
- "voltage" → "Line Voltage" (not "RYB Phase Voltage")
- "ryb voltage" → "RYB Phase Voltage"
- "frequency" → "Frequency"

Return JSON:
{{
    "matched_param": "<exact parameter name from list>" or null,
    "confidence": <0.0 to 1.0>,
    "reasoning": "brief explanation",
    "is_ambiguous": false,
    "alternatives": []
}}
"""
            response = openai_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0,
                max_tokens=150
            )
            result = json.loads(response.choices[0].message.content)
            # ✅ ADD THIS - Handle ambiguity
            # ✅ Handle ambiguity
            is_ambiguous = result.get('is_ambiguous', False)
            alternatives = result.get('alternatives', [])
            
            if is_ambiguous and alternatives and len(alternatives) > 1:
                logging.warning(f"   ⚠️ Ambiguous parameter match: {alternatives}")
                return {
                    'ambiguous': True,
                    'alternatives': alternatives,
                    'params': None,
                    'pin': None
                }
            
            matched_name = result.get('matched_param')
            confidence = result.get('confidence', 0.0)
            reasoning = result.get('reasoning', '')
            
            logging.info(f"   🤖 LLM Match: '{matched_name}' (confidence: {confidence:.2f})")
            logging.info(f"   Reasoning: {reasoning}")

            if matched_name and confidence > 0.6:
                for p in available_params:
                    if p['params'] == matched_name:
                        return p
        except Exception as e:
            logging.error(f"   ❌ LLM Fuzzy matching failed: {e}")

    # No match
    return None


# ==================== DATA EXTRACTION ====================

def extract_parameter_values(device_id: int, pin: str, time_info: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """
    Extract values for a specific pin from device_data/historical tables.
    
    Args:
        device_id: Device ID
        pin: Pin number (e.g., 'V20')
        time_info: Time range dict
    
    Returns: DataFrame with columns [timestamp, value] or None
    """
    logging.info(f"\n{'='*80}")
    logging.info(f"📊 EXTRACTING PARAMETER VALUES")
    logging.info(f"{'='*80}")
    logging.info(f"Device ID: {device_id}")
    logging.info(f"Pin: {pin}")
    logging.info(f"Time range: {time_info.get('start_time')} to {time_info.get('end_time')}")
    
    tables = get_tables_for_time_range(time_info)
    if not tables:
        logging.error("❌ No tables found for time range")
        logging.info(f"{'='*80}\n")
        return None
    
    logging.info(f"📋 Querying {len(tables)} tables")
    
    all_data = []
    
    for idx, table in enumerate(tables, 1):
        logging.info(f"[{idx}/{len(tables)}] Querying {table}...")
        
        json_column = 'device_value' if table == "device_data" else 'raw_value'
        time_column = 'datetime' if table == "device_data" else 'updatedtime'
        
        sql = f"""
        SELECT 
            {time_column} as timestamp,
            JSON_UNQUOTE({json_column}->'$.{pin}') as value
        FROM {table}
        WHERE 
            device_id = :device_id
            AND {time_column} BETWEEN :start AND :end
        ORDER BY {time_column} ASC
        """
        
        params = {
            'device_id': device_id,
            'start': time_info['start_time'],
            'end': time_info['end_time']
        }
        
        try:
            with engine.connect() as conn:
                result = conn.execute(text(sql), params).fetchall()
            
            if result:
                df = pd.DataFrame(result, columns=['timestamp', 'value'])
                all_data.append(df)
                logging.info(f"   ✅ Fetched {len(df)} rows")
            else:
                logging.info(f"   ⚠️  No data")
        
        except Exception as e:
            if "doesn't exist" not in str(e):
                logging.error(f"   ❌ Error: {e}")
            else:
                logging.warning(f"   ⚠️  Table doesn't exist (skipping)")
            continue
    
    if not all_data:
        logging.error("❌ No data found in any table")
        logging.info(f"{'='*80}\n")
        return None
    
    df_combined = pd.concat(all_data, ignore_index=True)
    df_combined['timestamp'] = pd.to_datetime(df_combined['timestamp'])
    df_combined['value'] = pd.to_numeric(df_combined['value'], errors='coerce')
    df_combined = df_combined.dropna(subset=['value'])
    df_combined = df_combined.sort_values('timestamp')
    
    logging.info(f"✅ Total valid data points: {len(df_combined)}")
    logging.info(f"{'='*80}\n")
    
    return df_combined


def calculate_parameter_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate statistics for parameter values."""
    if df is None or df.empty:
        return None
    
    stats = {
        'average': float(df['value'].mean()),
        'minimum': float(df['value'].min()),
        'maximum': float(df['value'].max()),
        'median': float(df['value'].median()),
        'data_points': len(df),
        'start_time': df['timestamp'].min().strftime('%Y-%m-%d %H:%M:%S'),
        'end_time': df['timestamp'].max().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    logging.info(f"📊 Statistics:")
    logging.info(f"   Average: {stats['average']:.2f}")
    logging.info(f"   Min: {stats['minimum']:.2f}")
    logging.info(f"   Max: {stats['maximum']:.2f}")
    logging.info(f"   Median: {stats['median']:.2f}")
    logging.info(f"   Data points: {stats['data_points']}")
    
    return stats

# ==================== MAIN HANDLER ====================

def handle_parameter_query(
    user_query: str,
    device_id: int,
    device_name: str,
    cluster_id: int,
    parameter_name: str,
    time_info: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Main handler for parameter queries.
    
    Returns:
        {
            'success': bool,
            'parameter': str,
            'pin': str,
            'statistics': {...},
            'message': str
        }
    """
    logging.info(f"\n{'='*80}")
    logging.info(f"🎯 PARAMETER QUERY HANDLER")
    logging.info(f"{'='*80}")
    logging.info(f"Query: {user_query}")
    logging.info(f"Device: {device_name} (ID: {device_id})")
    logging.info(f"Cluster ID: {cluster_id}")
    logging.info(f"Parameter: {parameter_name}")
    
    # Default time range: Last 2 months
    if not time_info or not time_info.get('start_time'):
        from datetime import datetime as dt  # Ensure local import
        end_time = dt.now()
        start_time = end_time - timedelta(days=60)
        time_info = {
            'start_time': start_time.strftime('%Y-%m-%dT%H:%M:%S'),
            'end_time': end_time.strftime('%Y-%m-%dT%H:%M:%S')
        }
        logging.info(f"⏱️  Using default time: Last 2 months")
    
    # Step 1: Resolve parameter to pin
    param_info = resolve_parameter_to_pin(cluster_id, device_id, parameter_name)
    
    # ✅ Check for ambiguity
    if param_info and param_info.get('ambiguous'):
        alternatives = param_info.get('alternatives', [])
        message = f"Multiple parameters match '{parameter_name}'. Please specify:\n\n"
        message += "\n".join(f"- {alt}" for alt in alternatives)
        logging.warning(f"⚠️ Ambiguous match")
        logging.info(f"{'='*80}\n")
        return {'success': False, 'message': message}
    
    if not param_info:
        # Get all available parameters for error message
        sql = """
        SELECT pin, name, address
        FROM datastream
        WHERE cluster_id = :cluster_id AND isdelete = 0
        """
        
        available_params = []
        try:
            with engine.connect() as conn:
                result = conn.execute(text(sql), {"cluster_id": cluster_id}).fetchall()
            
            for row in result:
                address_json = row[2]
                if address_json:
                    try:
                        address_array = json.loads(address_json)
                        for addr_obj in address_array:
                            if isinstance(addr_obj, dict):
                                for key in addr_obj.keys():
                                    if key.startswith('address-'):
                                        addr_data = addr_obj[key]
                                        if isinstance(addr_data, dict):
                                            devices = addr_data.get('devices', [])
                                            if device_id in devices:
                                                params = addr_data.get('params')
                                                if params:
                                                    available_params.append(params)
                    except:
                        pass
        except:
            pass
        
        available_msg = ""
        if available_params:
            available_msg = f"\n\nAvailable parameters for {device_name}:\n" + "\n".join(f"- {p}" for p in available_params[:10])
        
        logging.error(f"❌ Parameter not found")
        logging.info(f"{'='*80}\n")
        
        # return {
        #     'success': False,
        #     'message': f"There is no parameter like '{parameter_name}' for {device_name}.{available_msg}"
        # }

        query_display = parameter_name if parameter_name else user_query
        return {
            'success': False,
            'message': f"There is no parameter like '{query_display}' for {device_name}.{available_msg}"
        }
    # Step 2: Extract data
    df = extract_parameter_values(device_id, param_info['pin'], time_info)
    
    if df is None or df.empty:
        logging.error(f"❌ No data found for {param_info['params']}")
        logging.info(f"{'='*80}\n")
        
        # ✅ Context-aware no-data message
        time_source = time_info.get('source', 'none')
        start_time = time_info.get('start_time', '')
        end_time = time_info.get('end_time', '')
        
        if time_source in ['default_param_2m', 'none']:
            message = f"No data found for {param_info['params']} ({param_info['pin']}) in the last 2 months (default range). Would you like to try a different time period?"
        else:
            try:
                from datetime import datetime
                start_str = datetime.fromisoformat(start_time).strftime('%b %d, %Y')
                end_str = datetime.fromisoformat(end_time).strftime('%b %d, %Y')
                message = f"No data found for {param_info['params']} ({param_info['pin']}) from {start_str} to {end_str}. Try a different time range?"
            except:
                message = f"No data found for {param_info['params']} ({param_info['pin']}) in the specified time range."
        
        return {'success': False, 'message': message}
    
    # Step 3: Calculate statistics
    stats = calculate_parameter_statistics(df)
    
    logging.info(f"✅ Parameter query completed successfully")
    logging.info(f"{'='*80}\n")
    
    return {
        'success': True,
        'parameter': param_info['params'],
        'pin': param_info['pin'],
        'statistics': stats,
        'device_name': device_name,
        'message': None
    }

# ==================== DEVICE MISSING HANDLER ====================

def get_available_devices_message(df_devices: pd.DataFrame) -> str:
    """Generate message with available devices when device is missing from query."""
    device_names = df_devices['device_name'].tolist()[:20]  # Show first 20
    
    message = "Please specify which device you're asking about.\n\nAvailable devices:\n"
    message += "\n".join(f"- {name}" for name in device_names)
    
    if len(df_devices) > 20:
        message += f"\n... and {len(df_devices) - 20} more"
    
    return message

def check_exit_command(user_message: str) -> bool:
    """Check if user wants to exit the current query flow."""
    exit_commands = ['exit', 'quit', 'stop', 'cancel', 'nevermind', 'never mind','hold','skip']
    return user_message.lower().strip() in exit_commands


# ==================== LIST PARAMETERS FEATURE ====================

def detect_list_parameters_query(query: str) -> bool:
    """Detect if user is asking for list of available parameters."""
    query_lower = query.lower()
    
    # More specific patterns
    indicators = [
        # Generic listing
        'what parameters', 'which parameters', 'show parameters', 'list parameters',
        'available parameters', 'parameters available', 'parameters present',

        # Pins listing
        'what pins', 'which pins', 'show pins', 'list pins',
        'available pins', 'pins available', 'pin list',

        # Common user ways of asking
        'give pins', 'give me pins', 'device pins', 'pins for', 'all pins',
        'give parameters', 'all parameters', 'parameter list'
    ]
    
    # Check for exact phrase matches
    for indicator in indicators:
        if indicator in query_lower:
            # ✅ Exclude if it's asking for VALUES (not list)
            if not any(val_kw in query_lower for val_kw in ['value', 'reading', 'data', 'for', 'last', 'past']):
                return True
    
    # Special case: "device parameters?" or "chiller parameters?"
    if re.search(r'(\w+)\s+(parameters?|pins?)\??s*$', query_lower):
        return True
    
    # NEW: Also match "list of X"
    if re.search(r'list\s+of\s+(parameters?|pins?)', query_lower):
        return True
    
    if re.search(r'\b(pins?|parameters?)\b', query_lower):
        return True
    
    return False

def get_available_parameters_for_device(cluster_id: int, device_id: int, device_name: str) -> str:
    """
    Get list of all available parameters/pins for a device.
    
    Returns formatted message with all parameters.
    """
    logging.info(f"\n{'='*80}")
    logging.info(f"📋 LISTING AVAILABLE PARAMETERS")
    logging.info(f"{'='*80}")
    logging.info(f"Device: {device_name} (ID: {device_id})")
    logging.info(f"Cluster ID: {cluster_id}")
    
    sql = """
    SELECT pin, name, address
    FROM datastream
    WHERE cluster_id = :cluster_id AND isdelete = 0
    ORDER BY pin
    """
    
    try:
        with engine.connect() as conn:
            result = conn.execute(text(sql), {"cluster_id": cluster_id}).fetchall()
        
        if not result:
            logging.warning(f"⚠️  No datastream found for cluster_id={cluster_id}")
            return f"No parameters found for {device_name}."
        
        all_params = []
        
        for row in result:
            pin = row[0]
            name = row[1]
            address_json = row[2]
            
            if not address_json:
                continue
            
            try:
                address_array = json.loads(address_json)
                
                if not isinstance(address_array, list):
                    continue
                
                for address_obj in address_array:
                    if not isinstance(address_obj, dict):
                        continue
                    
                    obj_pin = address_obj.get('pin')
                    if obj_pin != pin:
                        continue
                    
                    address_keys = sorted([k for k in address_obj.keys() if k.startswith('address-')])
                    
                    for addr_key in address_keys:
                        addr_data = address_obj[addr_key]
                        
                        if not isinstance(addr_data, dict):
                            continue
                        
                        devices_list = addr_data.get('devices', [])
                        params_name = addr_data.get('params')
                        
                        if isinstance(devices_list, list) and device_id in devices_list and params_name:
                            all_params.append(f"{params_name} ({pin})")
                            break
            
            except json.JSONDecodeError:
                continue
        
        if not all_params:
            logging.warning(f"⚠️  No parameters found for device_id={device_id}")
            return f"No parameters configured for {device_name}."
        
        # Remove duplicates
        all_params = sorted(list(set(all_params)))
        
        message = f"Available parameters for {device_name}:\n\n"
        message += "\n".join(f"• {param}" for param in all_params)
        
        logging.info(f"✅ Found {len(all_params)} parameters")
        logging.info(f"{'='*80}\n")
        
        return message
        
    except Exception as e:
        logging.error(f"❌ Failed to get parameters: {e}")
        return f"Error retrieving parameters for {device_name}."
