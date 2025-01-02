from pathlib import Path

# Get the absolute path to the project root
PROJECT_ROOT = Path(__file__).resolve().parent
APP_ROOT = PROJECT_ROOT.parent.parent.parent
print(f"Extra app root: {APP_ROOT}")
# Add paths that need to be in sys.path
EXTRA_PATHS = [
    str(APP_ROOT),  # For importing from app/
]