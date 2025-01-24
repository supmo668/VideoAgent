"""
Tracing configuration for LangSmith observability.
"""
import os
from contextlib import contextmanager
from typing import Optional
from langsmith import Client
from dotenv import load_dotenv

load_dotenv()

class TracingConfig:
    """Configuration for LangSmith tracing."""
    
    def __init__(self):
        """Initialize tracing configuration."""
        self.api_key = os.getenv("LANGCHAIN_API_KEY")
        self.api_url = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
        self.project = os.getenv("LANGCHAIN_PROJECT", "default")
        
        if not self.api_key:
            raise ValueError("LANGCHAIN_API_KEY must be set in environment variables")
        
        # Initialize LangSmith client
        self.client = Client(
            api_url=self.api_url,
            api_key=self.api_key,
        )
    
    @contextmanager
    def trace_context(self, project_name: Optional[str] = None):
        """
        Context manager for tracing a block of code.
        
        Args:
            project_name: Optional project name to use for this trace
        """
        original_project = os.getenv("LANGCHAIN_PROJECT")
        if project_name:
            os.environ["LANGCHAIN_PROJECT"] = project_name
        
        try:
            yield
        finally:
            if original_project:
                os.environ["LANGCHAIN_PROJECT"] = original_project
            else:
                os.environ.pop("LANGCHAIN_PROJECT", None)
    
    def enable_tracing(self):
        """Enable tracing globally."""
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
    
    def disable_tracing(self):
        """Disable tracing globally."""
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
