import yaml

def check_login(username, password):
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    for user in config["users"]:
        if user["username"] == username and user["password"] == password:
            return True
    return False