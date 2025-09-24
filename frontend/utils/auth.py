# auth_utils.py

import json
from werkzeug.security import generate_password_hash, check_password_hash

USERS_FILE = "users.json"

def load_users():
    try:
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)

def register_user(email, password, name):
    users = load_users()
    if email in users:
        return False, "User already exists."
    hashed_pw = generate_password_hash(password)
    users[email] = {"name": name, "password": hashed_pw}
    save_users(users)
    return True, "Registered successfully!"

def login_user(email, password):
    users = load_users()
    if email not in users:
        return False, "User not found."
    if not check_password_hash(users[email]["password"], password):
        return False, "Incorrect password."
    return True, users[email]["name"]