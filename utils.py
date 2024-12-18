import yaml

from dotenv import load_dotenv
load_dotenv()

def load_config(config_path: str):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg