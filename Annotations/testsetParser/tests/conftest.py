import pytest
import os
import tempfile
import shutil

from dotenv import load_dotenv
load_dotenv()

@pytest.fixture(scope="session")
def temp_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture(autouse=True)
def setup_environment():
    """Setup test environment variables."""
    original_env = dict(os.environ)
    os.environ["HUGGINGFACE_TOKEN"] = os.getenv("HUGGINGFACE_TOKEN")
    assert os.getenv("HUGGINGFACE_TOKEN")
    yield
    os.environ.clear()
    os.environ.update(original_env)
