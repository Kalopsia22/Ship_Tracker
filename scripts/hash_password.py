#!/usr/bin/env python3
"""Create a PBKDF2 hash for AUTH_USERS_JSON. Password is prompted, never echoed or stored."""
import getpass
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from maritime.security import Role, hash_password  # noqa: E402

user = input("Username: ").strip()
role = input("Role (viewer/analyst/admin): ").strip()
Role.parse(role)
pw = getpass.getpass("Password (min 8 chars): ")
if pw != getpass.getpass("Repeat: "):
    sys.exit("Passwords differ")
print("\nMerge this into AUTH_USERS_JSON (or .streamlit/secrets.toml):\n")
print(json.dumps({user: {"password_hash": hash_password(pw), "role": role.lower()}}))
