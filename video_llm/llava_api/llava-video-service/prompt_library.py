"""
Prompt Library Module for managing and retrieving prompts from database.
"""
import os
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
from langsmith import traceable
from logging_config import get_logger

load_dotenv()
logger = get_logger()

class PromptLibrary:
    """Manager for storing and retrieving prompts from database."""
    
    def __init__(self):
        """Initialize database connection if possible."""
        self.conn = None
        try:
            conn_uri = os.getenv("DATABASE_CONN_URI")
            if conn_uri:
                self.conn = psycopg2.connect(conn_uri, sslmode='require')
                self.conn.autocommit = True
            logger.info("Connected to Prompt Library database successfully.")
        except Exception as e:
            logger.warning(f"Failed to initialize PromptLibrary: {str(e)}")

    @traceable(run_type="chain")
    async def get_prompt(self, name: str) -> str:
        """
        Fetch the prompt content from the prompts table by name, filtering for format_instruction type.

        Args:
            name: The unique name of the prompt to fetch

        Returns:
            The prompt text if found, empty string otherwise
        """
        if not self.conn:
            logger.warning("Database connection not available")
            return ""
        
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                query = """
                    SELECT prompt 
                    FROM prompts 
                    WHERE name = %s 
                    AND type = 'format_instruction'
                """
                logger.debug(f"Executing query: {query} with params: ({name},)")
                cur.execute(query, (name,))
                result = cur.fetchone()

                if not result:
                    logger.warning(f"No prompt found with name '{name}' and type 'format_instruction'")
                    return ""

                prompt = result["prompt"]
                logger.info(f"Successfully retrieved prompt: {prompt[:50]}...")
                return prompt
                
        except Exception as e:
            logger.error(f"Error fetching prompt from database: {str(e)}")
            return ""

    @traceable(run_type="chain")
    async def list_prompts(self, category: str = None):
        """
        List all available format_instruction prompts, optionally filtered by category.

        Args:
            category: Optional category to filter prompts

        Returns:
            List of prompt names
        """
        if not self.conn:
            logger.warning("Database connection not available")
            return []
            
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                query = "SELECT name FROM prompts WHERE type = 'format_instruction'"
                params = ()
                
                if category:
                    query += " AND category = %s"
                    params = (category,)

                cur.execute(query, params)
                prompts = [row["name"] for row in cur.fetchall()]
                logger.info(f"Found {len(prompts)} format_instruction prompts")
                return prompts
        except Exception as e:
            logger.error(f"Error listing prompts: {str(e)}")
            return []

    def __del__(self):
        """Close database connection on cleanup."""
        if self.conn:
            try:
                self.conn.close()
            except:
                pass
