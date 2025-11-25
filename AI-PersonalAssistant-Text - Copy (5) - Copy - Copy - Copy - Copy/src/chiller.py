# src/chiller.py
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple
from sqlalchemy import text

from config import TARIFF_RATE
from database import engine, get_tables_for_time_range
from utils import get_efficiency_rating, _get_efficiency_note


# ----------------------------------------------------------------------
# COPIED FROM utils.py → _estimate_energy_from_panel
# ----------------------------------------------------------------------
def _estimate_energy_from_panel(df: pd.DataFrame, device_id: int, time_info: Dict[str, Any]) -> tuple:
    logging.info(f"\n{'='*60}")
    logging.info(f"PANEL-BASED ENERGY CALCULATION START")
    logging.info(f"Device ID: {device_id} (Chiller)")
    logging.info(f"Panel Device: 1211")
    logging.info(f"{'='*60}")
    
    if df.empty or len(df) < 2:
        logging.warning(f"Insufficient data points: {len(df)}")
        logging.info(f"{'='*60}\n")
        return 0.0, 0.0, 0
    
    time_column = 'datetime' if 'datetime' in df.columns else 'updatedtime'
    df = df.sort_values(time_column).copy()
    
    df['device_time'] = pd.to_datetime(df[time_column])
    df['device_time_rounded'] = df['device_time'].dt.floor('min')
    
    timestamps = df['device_time_rounded'].unique()
    timestamps = sorted(timestamps)
    
    logging.info(f"Chiller ON data points: {len(timestamps)}")
    logging.info(f"Time range: {timestamps[0]} to {timestamps[-1]}")
    
    intervals = []
    for i in range(len(timestamps) - 1):
        start_time = timestamps[i]
        end_time = timestamps[i + 1]
        intervals.append((start_time, end_time))
    
    logging.info(f"Total intervals to process: {len(intervals)}")
    
    tables = get_tables_for_time_range(time_info)
    if not tables:
        logging.warning(f"No tables available for panel device")
        logging.info(f"{'='*60}\n")
        return 0.0, 0.0, 0
    
    logging.info(f"Fetching all panel data for time range...")
    
    panel_data = []
    
    for table in tables:
        json_column = 'device_value' if table == "device_data" else 'raw_value'
        time_column_panel = 'datetime' if table == "device_data" else 'updatedtime'
        
        sql = f"""
        SELECT {time_column_panel} as timestamp, {json_column}
        FROM {table}
        WHERE device_id = 1211
        AND {time_column_panel} >= :start_time
        AND {time_column_panel} <= :end_time
        ORDER BY {time_column_panel} ASC
        """
        
        try:
            with engine.connect() as conn:
                result = conn.execute(text(sql), {
                    'start_time': df[time_column].min().strftime('%Y-%m-%d %H:%M:%S'),
                    'end_time': df[time_column].max().strftime('%Y-%m-%d %H:%M:%S')
                }).fetchall()
            
            if result:
                for row in result:
                    try:
                        json_str = row[1]
                        if json_str:
                            json_data = json.loads(json_str) if isinstance(json_str, str) else json_str
                            if isinstance(json_data, dict):
                                v1_value = float(json_data.get('V1', 0))
                                timestamp = pd.to_datetime(row[0]).floor('min')
                                panel_data.append({'timestamp': timestamp, 'V1': v1_value})
                    except (json.JSONDecodeError, ValueError, TypeError) as e:
                        logging.warning(f"   Failed to parse row: {e}")
                        continue
                
                logging.info(f"   Fetched {len(panel_data)} panel readings from {table}")
        
        except Exception as e:
            if "doesn't exist" not in str(e):
                logging.error(f"   Error querying {table}: {e}")
            continue
    
    if not panel_data:
        logging.warning(f"No panel data found")
        logging.info(f"{'='*60}\n")
        return 0.0, 0.0, 0
    
    panel_df = pd.DataFrame(panel_data)
    panel_df = panel_df.drop_duplicates(subset=['timestamp']).set_index('timestamp').sort_index()
    
    logging.info(f"Panel DataFrame ready: {len(panel_df)} unique timestamps")
    
    total_energy_kwh = 0.0
    intervals_processed = 0
    
    for idx, (start_time, end_time) in enumerate(intervals, 1):
        logging.info(f"\n--- Interval {idx}/{len(intervals)} ---")
        logging.info(f"Chiller ON: {start_time} to {end_time}")
        
        v1_start = None
        v1_end = None
        
        try:
            mask = (panel_df.index >= start_time) & (panel_df.index <= end_time)
            interval_data = panel_df[mask]
            
            if not interval_data.empty:
                valid_data = []
                for ts in interval_data.index:
                    if (ts.year == start_time.year and 
                        ts.month == start_time.month and 
                        ts.day == start_time.day and
                        start_time.hour <= ts.hour <= end_time.hour):
                        valid_data.append(ts)
                
                if valid_data:
                    interval_data = interval_data.loc[valid_data]
                else:
                    interval_data = pd.DataFrame()
            
            if not interval_data.empty and len(interval_data) >= 2:
                v1_start = interval_data['V1'].iloc[0]
                v1_end = interval_data['V1'].iloc[-1]
                
                logging.info(f"   Panel data: {interval_data.index[0]} to {interval_data.index[-1]}")
                logging.info(f"   V1 values: {v1_start:.2f} to {v1_end:.2f}")

            elif not interval_data.empty and len(interval_data) == 1:
                logging.warning(f"   Only 1 panel data point found at {interval_data.index[0]}")
                continue
            else:
                logging.warning(f"   No panel data found for this interval")
                continue
                
        except Exception as e:
            logging.warning(f"   Failed to find panel data: {e}")
            continue
        
        consumption_diff = v1_end - v1_start
        
        if consumption_diff <= 0:
            logging.warning(f"   Subtraction: {consumption_diff:.2f}")
            logging.warning(f"   Skipping (negative or zero consumption)")
            continue
        
        consumption_kwh = consumption_diff / 1000
        
        logging.info(f"   Subtraction: {consumption_diff:.2f}")
        logging.info(f"   Energy: {consumption_kwh:.3f} kWh")
        logging.info(f"   Cost: ₹{consumption_kwh * TARIFF_RATE:.2f}")
        
        total_energy_kwh += consumption_kwh
        intervals_processed += 1

    total_cost_inr = total_energy_kwh * TARIFF_RATE
    
    logging.info(f"\n{'='*60}")
    logging.info(f"PANEL-BASED ENERGY CALCULATION COMPLETE")
    logging.info(f"   Total intervals: {len(intervals)}")
    logging.info(f"   Processed intervals: {intervals_processed}")
    logging.info(f"   Skipped intervals: {len(intervals) - intervals_processed}")
    logging.info(f"   Total Energy: {total_energy_kwh:.3f} kWh")
    logging.info(f"   Total Cost: ₹{total_cost_inr:.2f}")
    logging.info(f"{'='*60}\n")
    
    return total_energy_kwh, total_cost_inr, intervals_processed


# ----------------------------------------------------------------------
# COPIED FROM utils.py → calculate_device_metrics_with_llm (CHILLER PART)
# ----------------------------------------------------------------------
def calculate_device_metrics_with_llm(df: pd.DataFrame, device_id: int, query: str, device_name: str, device_type: int = 1, time_info: Dict[str, Any] = None) -> Dict[str, Any]: 
    logging.info(f"\n{'='*60}")
    logging.info(f"CALCULATING METRICS FOR DEVICE {device_id}")
    logging.info(f"{'='*60}")
    
    time_column = 'datetime' if 'datetime' in df.columns else 'updatedtime'
    df = df.sort_values(time_column).drop_duplicates(subset=[time_column], keep='first')
    
    logging.info(f"Initial rows: {len(df)}")
    
    if 'device_value' in df.columns:
        logging.debug("Extracting V18, V19, V20 from device_value...")
        df['power'] = df['device_value'].apply(lambda x: json.loads(x).get('V18') if x else None)
        df['temp_in'] = df['device_value'].apply(lambda x: json.loads(x).get('V19') if x else None)
        df['temp_out'] = df['device_value'].apply(lambda x: json.loads(x).get('V20') if x else None)
    elif 'raw_value' in df.columns:
        logging.debug("Extracting V18, V19, V20 from raw_value...")
        df['power'] = df['raw_value'].apply(lambda x: json.loads(x).get('V18') if x else None)
        df['temp_in'] = df['raw_value'].apply(lambda x: json.loads(x).get('V19') if x else None)
        df['temp_out'] = df['raw_value'].apply(lambda x: json.loads(x).get('V20') if x else None)
    
    df['power'] = pd.to_numeric(df['power'], errors='coerce')
    df['temp_in'] = pd.to_numeric(df['temp_in'], errors='coerce')
    df['temp_out'] = pd.to_numeric(df['temp_out'], errors='coerce')
    df = df.dropna(subset=['power', 'temp_in', 'temp_out'])
    
    if df.empty:
        logging.warning(f"No valid data after filtering")
        logging.info(f"{'='*60}\n")
        return {
            'device_id': device_id,
            'error': 'No valid data',
            'cop': None,
            'eer': None,
            'cooling_capacity_tr': 0,
            'average_power_kw': 0,
            'energy_cost_inr': 0,
            'data_points': 0,
            'hours_analyzed': 0,
            'efficiency_rating': 'N/A'
        }
    
    logging.info(f"Valid rows after filtering: {len(df)}")

    if device_type == 1:
        logging.info(f"Using panel-based energy calculation (device_type = 1)")
        
        if time_info:
            consumption_kwh, energy_cost, intervals_processed = _estimate_energy_from_panel(df, device_id, time_info)
        else:
            logging.warning(f"No time_info provided, falling back to old method")
            df['instantaneous_power_kw'] = (1.732 * 405 * df['power'] * 0.9) / 1000
            consumption_kwh = np.sum(df['instantaneous_power_kw'].diff().fillna(0).clip(lower=0)) / 60
            energy_cost = consumption_kwh * TARIFF_RATE
        
        df['instantaneous_power_kw'] = (1.732 * 405 * df['power'] * 0.9) / 1000
        avg_power = df['instantaneous_power_kw'].mean()
        
        hours = (df.iloc[-1][time_column] - df.iloc[0][time_column]).total_seconds() / 3600
        
        logging.info(f"Time span: {hours:.2f} hours")
        logging.info(f"Average power (from V18): {avg_power:.2f} kW")
    
    else:
        logging.warning(f"Energy calculation not implemented for device_type = {device_type}")
        consumption_kwh = 0.0
        energy_cost = 0.0
        avg_power = 0.0
        hours = 0.0
        
        logging.info(f"Skipping energy calculation for non-chiller device")

    df["delta_T"] = (df["temp_out"] - df["temp_in"]) / 1.8
    mean_delta_t = df["delta_T"].mean()

    logging.info(f"Average Delta T: {mean_delta_t:.2f} degrees C")

    if df["delta_T"].notnull().any() and abs(mean_delta_t) > 0:
        cooling_kw = (54.5 * 1000 * 4.186 * abs(mean_delta_t)) / 3600
        cooling_tr = cooling_kw / 3.517
        cop = (cooling_kw / avg_power) if avg_power > 0 else None
        eer = (cooling_kw * 3.412) / avg_power if avg_power > 0 else None
        
        logging.info(f"Cooling capacity: {cooling_tr:.2f} TR ({cooling_kw:.2f} kW)")
        logging.info(f"COP: {cop:.3f}" if cop else "COP: N/A")
        logging.info(f"EER: {eer:.3f}" if eer else "EER: N/A")
    else:
        cooling_kw = cooling_tr = 0
        cop = eer = None
        logging.warning(f"Cannot calculate COP/EER (insufficient temperature data)")

    cop_str = f"{cop:.2f}" if cop is not None else 'N/A'
    eer_str = f"{eer:.2f}" if eer is not None else 'N/A'
    efficiency_rating = get_efficiency_rating(cop) if cop else "N/A"
    logging.info(f"Efficiency rating: {efficiency_rating}")

    run_time_hours = 0.0
    if device_type == 1 and time_info:
        try:
            if 'intervals_processed' in locals():
                run_time_hours = intervals_processed / 60.0
            else:
                run_time_hours = len(df) / 60.0
        except:
            run_time_hours = hours

    metrics = {
        "device_id": device_id,
        "energy_consumed_kwh": consumption_kwh,
        "energy_cost_inr": consumption_kwh * TARIFF_RATE,
        "cooling_capacity_tr": cooling_tr,
        "average_power_kw": avg_power,
        "cop": cop,
        "eer": eer,
        "data_points": len(df),
        "hours_analyzed": hours,
        "run_time_hours": run_time_hours,
        "efficiency_rating": efficiency_rating,
        "efficiency_note": _get_efficiency_note(efficiency_rating)
    }
    
    logging.info(f"Metrics calculation complete")
    logging.info(f"{'='*60}\n")
    return metrics