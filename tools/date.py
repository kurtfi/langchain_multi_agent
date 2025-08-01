"""
Date tool for getting the current date and time.
This tool is safe and can be used for any time-based queries.
"""

from langchain_core.tools import tool
from datetime import datetime
import logging

# Configure logging
logger = logging.getLogger(__name__)

@tool
def get_current_date() -> str:
    """
    Returns the current date and time.
    
    This tool is useful for any time-based queries or when you need to know the current date.
    
    Returns:
        str: A formatted string with the current date
    """
    try:
        current_date = datetime.now().strftime('%d %B %Y')
        result = f"The current date is: {current_date}"
        logger.info("Current date retrieved successfully")
        return result
    except Exception as e:
        logger.error(f"Error getting current date: {str(e)}")
        return f"Error retrieving current date: {str(e)}"
