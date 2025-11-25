# # auth.py
# import json
# import logging
# from pathlib import Path
# from typing import Optional, Dict

# # Load users from JSON file
# SCRIPT_DIR = Path(__file__).parent.resolve()
# USERS_FILE = SCRIPT_DIR / "vertical_users.json"

# def load_users() -> list:
#     """Load user credentials from JSON file."""
#     try:
#         with open(USERS_FILE, 'r') as f:
#             data = json.load(f)
#             return data.get('users', [])
#     except FileNotFoundError:
#         logging.error(f"❌ {USERS_FILE} not found!")
#         return []
#     except json.JSONDecodeError as e:
#         logging.error(f"❌ Invalid JSON in {USERS_FILE}: {e}")
#         return []

# def authenticate_user(username: str, password: str) -> Optional[Dict]:
#     """
#     Authenticate user against JSON file.
#     Returns user data if valid, None otherwise.
#     """
#     logging.info(f"🔐 Authentication attempt: {username}")
    
#     users = load_users()
    
#     for user in users:
#         if user['username'] == username and user['password'] == password:
#             logging.info(f"✅ Authentication successful: {username} → Vertical: {user['vertical_name']} (ID: {user['vertical_id']})")
#             return {
#                 "username": user['username'],
#                 "vertical_id": user['vertical_id'],
#                 "vertical_name": user['vertical_name']
#             }
    
#     logging.warning(f"❌ Invalid credentials: {username}")
#     return None

import logging
from typing import Optional, Dict, List

# 🔹 Define users directly as a Python list (no JSON file)
USERS: List[Dict] = [
    {
        "username": "niraltekuser",
        "password": "user@123",
        "vertical_id": 139,
        "vertical_name": "Niraltek"
    },
    {
        "username": "bigfootuser",
        "password": "user@123",
        "vertical_id": 161,
        "vertical_name": "BigFoot"
    },
    {
        "username": "ssncollegeuser",
        "password": "user@123",
        "vertical_id": 162,
        "vertical_name": "SSN College"
    },
    {
        "username": "hinehydraulicsuser",
        "password": "user@123",
        "vertical_id": 163,
        "vertical_name": "Hine Hydraulics"
    },
    {
        "username": "swamifeedsuser",
        "password": "user@123",
        "vertical_id": 167,
        "vertical_name": "Swami Feeds Pvt Ltd"
    },
    {
        "username": "mediscanuser",
        "password": "user@123",
        "vertical_id": 169,
        "vertical_name": "MediScan"
    },
    {
        "username": "mysticaluser",
        "password": "user@123",
        "vertical_id": 170,
        "vertical_name": "Mystical ColdStorage"
    },
    {
        "username": "madurasteelsuser",
        "password": "user@123",
        "vertical_id": 171,
        "vertical_name": "MADURA STEELS"
    },
    {
        "username": "srivariuser",
        "password": "user@123",
        "vertical_id": 172,
        "vertical_name": "Srivari Alloys"
    },
    {
        "username": "seoulstoreuser",
        "password": "user@123",
        "vertical_id": 173,
        "vertical_name": "SEOUL STORE"
    },
    {
        "username": "gunasolaruser",
        "password": "user@123",
        "vertical_id": 174,
        "vertical_name": "Guna Solar Project"
    },
    {
        "username": "seasonssuitesuser",
        "password": "user@123",
        "vertical_id": 176,
        "vertical_name": "Seasons Suites"
    },
    {
        "username": "thailanduser",
        "password": "user@123",
        "vertical_id": 177,
        "vertical_name": "Thailand"
    }
]


def authenticate_user(username: str, password: str) -> Optional[Dict]:
    """
    Authenticate user against hardcoded list.
    Returns user data if valid, None otherwise.
    """
    logging.info(f"🔐 Authentication attempt: {username}")
    
    for user in USERS:
        if user["username"] == username and user["password"] == password:
            logging.info(f"✅ Authentication successful: {username} → {user['vertical_name']} (ID: {user['vertical_id']})")
            return {
                "username": user["username"],
                "vertical_id": user["vertical_id"],
                "vertical_name": user["vertical_name"]
            }
    
    logging.warning(f"❌ Invalid credentials: {username}")
    return None
