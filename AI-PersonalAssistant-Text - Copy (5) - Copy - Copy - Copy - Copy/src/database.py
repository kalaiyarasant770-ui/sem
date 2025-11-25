# database.py
import re
import logging
import time
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from sqlalchemy import create_engine, text
from typing import Dict, Any, List, Optional
from config import DATABASE_URL, FORBIDDEN_KEYWORDS

# Initialize database engine with detailed logging
t_db_start = time.time()
logging.info("🔄 Initializing database connection...")
logging.info(f"   Host: {DATABASE_URL.split('@')[1].split('/')[0] if '@' in DATABASE_URL else 'unknown'}")

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=3600
)
logging.info(f"✅ Database engine initialized in {time.time() - t_db_start:.2f} seconds")
def validate_query_safety(query: str) -> Dict[str, Any]:
    """Check if query contains harmful or forbidden operations with context awareness."""
    logging.debug(f"🔍 Validating query safety: {query[:80]}...")
    query_lower = query.lower().strip()
    
    data_modification_keywords = [
        'delete', 'drop', 'truncate', 'remove', 'clear',
        'insert', 'create', 'replace', 'merge', 'set'
    ]
    
    advisory_indicators = [
        'how', 'what', 'when', 'why', 'should', 'can i', 'best way',
        'maintenance', 'parts', 'schedule', 'procedure', 'process',
        'component', 'system', 'equipment', 'tips', 'advice',
        'recommendation', 'practice', 'strategy', 'method', 'compressor',
        'condenser', 'evaporator', 'refrigerant', 'improve', 'optimize',
        'cop', 'efficiency', 'performance'
    ]
    
    has_advisory_context = any(indicator in query_lower for indicator in advisory_indicators)
    
    for keyword in data_modification_keywords:
        if re.search(rf'\b{keyword}\b', query_lower):
            logging.warning(f"🚫 Blocked forbidden operation: '{keyword}' in query: {query[:80]}...")
            return {
                "is_safe": False,
                "keyword": keyword,
                "message": f"I can only read and analyze chiller data, not {keyword} it. I have read-only access to ensure data integrity. Would you like to view the data instead?"
            }
    
    if re.search(r'\b(update|modify|change|alter)\b', query_lower):
        data_context_indicators = [
            'record', 'database', 'table', 'data', 'value', 'entry',
            'row', 'column', 'field', 'sql', 'query'
        ]
        
        has_data_context = any(indicator in query_lower for indicator in data_context_indicators)
        
        if has_data_context and not has_advisory_context:
            keyword = 'update' if 'update' in query_lower else ('modify' if 'modify' in query_lower else ('alter' if 'alter' in query_lower else 'change'))
            logging.warning(f"🚫 Blocked data modification: '{keyword}' in database context")
            return {
                "is_safe": False,
                "keyword": keyword,
                "message": f"I can only read and analyze chiller data, not {keyword} it. I have read-only access to ensure data integrity. Would you like to view the data instead?"
            }
        
        if has_advisory_context:
            logging.info(f"✅ Allowing '{query[:80]}...' - detected as advisory question (contains: {[i for i in advisory_indicators if i in query_lower][:3]})")
    
    logging.debug("✅ Query passed safety validation")
    return {"is_safe": True}


def validate_generated_sql(sql: str, user_query: str) -> Dict[str, Any]:
    """Enhanced validation that supports both single-device and UNION ALL multi-device queries."""
    logging.debug(f"🔍 Validating SQL query structure...")
    errors = []
    query_upper = sql.upper().strip()
    
    is_union_query = 'UNION ALL' in query_upper
    
    if is_union_query:
        logging.debug("   Detected UNION ALL query (multi-device)")
        union_parts = sql.split('UNION ALL')
        logging.debug(f"   Found {len(union_parts)} UNION parts")
        
        for i, part in enumerate(union_parts):
            part_upper = part.strip().upper()
            
            if not part_upper.startswith('(SELECT') and not part_upper.startswith('SELECT'):
                errors.append(f"UNION part {i+1} must be a SELECT statement")
            
            for keyword in FORBIDDEN_KEYWORDS:
                if re.search(rf'\b{keyword}\b', part_upper):
                    errors.append(f"Forbidden keyword '{keyword}' in UNION part {i+1}")
            
            valid_tables = ['device_data', 'historical_data_minute'] + [
                f"historical_data_minute_{y}_{m:02d}" for y in range(2020, 2026) for m in range(1, 13)
            ]
            table_found = any(table in part.lower() for table in valid_tables)
            if not table_found:
                errors.append(f"Invalid table in UNION part {i+1}")
            
            if 'device_id' not in part.lower():
                errors.append(f"UNION part {i+1} must filter by device_id")
            
            if re.search(r'[\'"]\d{4}-\d{2}-\d{2}[\'"]', part):
                errors.append(f"UNION part {i+1} must use :start and :end parameters, not inline dates")
        
    else:
        logging.debug("   Single-device query")
        if not query_upper.startswith('SELECT'):
            errors.append("Query must be a SELECT statement")
        
        for keyword in FORBIDDEN_KEYWORDS:
            if re.search(rf'\b{keyword}\b', query_upper):
                errors.append(f"Forbidden keyword '{keyword}' detected")
        
        valid_tables = ['device_data', 'historical_data_minute'] + [
            f"historical_data_minute_{y}_{m:02d}" for y in range(2020, 2026) for m in range(1, 13)
        ]
        table_found = any(table in sql.lower() for table in valid_tables)
        if not table_found:
            errors.append("Invalid or missing table name")
        
        if 'device_id = :device_id' not in sql.lower():
            errors.append("Query must filter by device_id using :device_id parameter")
        
        invalid_id_pattern = r"\bdevice_id\s*=\s*['\"`]?\d+['\"`]?"
        if re.search(invalid_id_pattern, sql, re.IGNORECASE):
            errors.append("Device_id must use :device_id parameter, not inline values")
        
        if re.search(r'[\'"]\d{4}-\d{2}-\d{2}[\'"]', sql):
            errors.append("Date values must use :start and :end parameters, not inline values")
    
    date_keywords = ['DATETIME', 'UPDATEDTIME', 'BETWEEN', 'DATE']
    time_indicators = ['last', 'past', 'today', 'yesterday', 'january', 'feb', 'month', 'week', 'day', 'hour', 'hrs']
    if any(kw in user_query.lower() for kw in time_indicators) and not any(kw in query_upper for kw in date_keywords):
        errors.append("Missing date filter for time-based query")
    
    if len(sql) > 2000:
        errors.append("Query too long (max 2000 characters)")
    
    is_valid = not errors
    query_type = "UNION ALL (multi-device)" if is_union_query else "single-device"
    
    if is_valid:
        logging.info(f"✅ SQL Validation [{query_type}]: Valid")
    else:
        logging.error(f"❌ SQL Validation [{query_type}]: Invalid - Errors: {errors}")
    
    return {"is_valid": is_valid, "errors": errors}


def safe_execute_sql(sql: str, params: Dict[str, Any], user_query: str) -> Dict[str, Any]:
    """Execute SQL safely with validation."""
    logging.debug(f"🔧 Executing SQL query...")
    logging.debug(f"   Params: {params}")
    
    validation = validate_generated_sql(sql, user_query)
    if not validation["is_valid"]:
        logging.warning(f"🚫 Blocked invalid SQL query")
        logging.warning(f"   Errors: {validation['errors']}")
        return {"error": "Invalid SQL query", "details": validation["errors"]}
    
    test_sql = f"{sql} LIMIT 1"
    try:
        logging.debug("   Testing query with LIMIT 1...")
        with engine.connect() as connection:
            test_result = connection.execute(text(test_sql), params).fetchall()
        logging.debug(f"   ✅ Test query succeeded")
    except Exception as e:
        logging.error(f"   ❌ Test SQL failed: {str(e)}")
        return {"error": f"Query failed: {str(e)}"}
    
    try:
        logging.debug("   Executing full query...")
        with engine.connect() as connection:
            result = connection.execute(text(sql), params).fetchall()
            df = pd.DataFrame(result)
        logging.info(f"✅ SQL executed successfully - Retrieved {len(df)} rows")
        return {"data": df}
    except Exception as e:
        logging.error(f"❌ SQL execution failed: {str(e)}")
        return {"error": f"Query execution failed: {str(e)}"}


def get_tables_for_time_range(time_info: Optional[Dict[str, Any]]) -> List[str]:
    """Determine which database tables to query based on time range."""
    logging.info(f"\n{'='*80}")
    logging.info(f"📊 TABLE SELECTION START")
    logging.info(f"{'='*80}")
    logging.info(f"Time info provided: {time_info}")
    
    tables = []
    current_date = datetime.now().date()
    current_month_start = current_date.replace(day=1)
    
    logging.info(f"📅 Current date: {current_date}")
    logging.info(f"📅 Current month start: {current_month_start}")
    
    if time_info is None or not isinstance(time_info, dict):
        logging.warning("❌ No valid time_info provided, defaulting to device_data for today")
        logging.info(f"{'='*80}\n")
        return ["device_data"]
    
    if 'last_hours' in time_info:
        try:
            hours = int(time_info['last_hours'])
            end = datetime.now()
            start = end - timedelta(hours=hours)
            start_date = start.date()
            end_date = end.date()
            logging.info(f"📊 Detected 'last_hours' format: {hours} hours")
        except (ValueError, TypeError):
            logging.error("❌ Invalid last_hours format, defaulting to device_data")
            logging.info(f"{'='*80}\n")
            return ["device_data"]
    else:
        start_date_str = time_info.get("start") or time_info.get("start_time")
        end_date_str = time_info.get("end") or time_info.get("end_time", start_date_str)
        
        if not start_date_str:
            logging.warning("❌ No valid start date provided, defaulting to device_data")
            logging.info(f"{'='*80}\n")
            return ["device_data"]
        
        try:
            if 'T' in start_date_str:
                start = datetime.strptime(start_date_str, "%Y-%m-%dT%H:%M:%S")
                start_date = start.date()
            else:
                start = datetime.strptime(start_date_str, "%Y-%m-%d")
                start_date = start.date()
            
            if end_date_str:
                if 'T' in end_date_str:
                    end = datetime.strptime(end_date_str, "%Y-%m-%dT%H:%M:%S")
                    end_date = end.date()
                else:
                    end = datetime.strptime(end_date_str, "%Y-%m-%d")
                    end_date = end.date()
            else:
                end_date = start_date
        except (ValueError, TypeError) as e:
            logging.error(f"❌ Invalid date format in time_info: {e}")
            logging.error(f"   start_date_str: {start_date_str}")
            logging.error(f"   end_date_str: {end_date_str}")
            logging.info(f"{'='*80}\n")
            return ["device_data"]
    
    logging.info(f"📅 Query date range: {start_date} to {end_date}")
    logging.info(f"{'-'*80}")
    
    # Check if query includes today
    if end_date >= current_date:
        tables.append("device_data")
        logging.info("✅ Adding device_data (query includes today)")
    
    # Include current month historical data
    if end_date >= current_month_start:
        if start_date < current_date:
            tables.append("historical_data_minute")
            logging.info("✅ Adding historical_data_minute (current month before today)")
    
    # Include past month tables
    if start_date < current_month_start:
        start_month = start_date.replace(day=1)
        actual_end_month = min(end_date, (current_month_start - timedelta(days=1)))
        end_month = actual_end_month.replace(day=1)
        
        logging.info(f"📊 Checking past months from {start_month} to {end_month}")
        
        current_iter = start_month
        while current_iter <= end_month:
            table_name = f"historical_data_minute_{current_iter.year}_{current_iter.month:02d}"
            tables.append(table_name)
            logging.info(f"✅ Adding {table_name} (past month)")
            current_iter = current_iter + relativedelta(months=1)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_tables = []
    for table in tables:
        if table not in seen:
            seen.add(table)
            unique_tables.append(table)
    
    logging.info(f"{'-'*80}")
    logging.info(f"📋 FINAL TABLE LIST ({len(unique_tables)} tables): {unique_tables}")
    logging.info(f"{'='*80}\n")
    return unique_tables