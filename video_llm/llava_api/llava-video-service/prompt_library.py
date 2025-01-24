"""
Prompt Library Module for managing and retrieving prompts from database.
"""
import os
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
from langsmith import traceable

load_dotenv()

class PromptLibrary:
    """Manager for storing and retrieving prompts from database."""
    
    def __init__(self):
        """Initialize database connection if possible."""
        self.conn = None
        try:
            conn_uri = os.getenv("DATABASE_CONN_URI")
            if conn_uri:
                self.conn = psycopg2.connect(conn_uri)
        except Exception as e:
            print(f"Warning: Failed to initialize PromptLibrary: {str(e)}")

    @traceable(run_type="chain")
    async def get_prompt(self, name: str) -> str:
        """
        Fetch the "prompt" field by name from the database.
        If multiple prompts exist with the same name, default to the one in the "schema" category.

        Args:
            name: The unique name of the prompt to fetch

        Returns:
            The prompt text if found, empty string otherwise
        """
        if not self.conn:
            return ""
        
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Fetch all prompts with the given name
            cur.execute("SELECT * FROM prompts WHERE name = %s", (name,))
            results = cur.fetchall()

            if not results:
                return ""

            # Default to "schema" category if multiple results exist
            selected_prompt = next((row for row in results if row.get("category") == "schema"), results[0])

            return selected_prompt.get("prompt", "")

    @traceable(run_type="chain")
    async def list_prompts(self, category: str = None):
        """
        List all available prompts, optionally filtered by category.

        Args:
            category: Optional category to filter prompts

        Returns:
            List of prompt names
        """
        if not self.conn:
            return []
            
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                query = "SELECT name FROM prompts"
                params = ()
                
                if category:
                    query += " WHERE category = %s"
                    params = (category,)

                cur.execute(query, params)
                return [row["name"] for row in cur.fetchall()]
        except Exception as e:
            print(f"Error listing prompts: {str(e)}")
            return []

    def __del__(self):
        """Close database connection on cleanup."""
        if self.conn:
            try:
                self.conn.close()
            except:
                pass
