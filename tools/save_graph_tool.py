from langchain_core.tools import tool
from datetime import datetime
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import contextlib

# Configure logging
logger = logging.getLogger(__name__)

@tool
def execute_and_save_graph_tool(code: str, stock_name: str) -> str:
    """
    Executes the given Python code to generate a graph and saves it as a JPEG file named after the stock.

    Args:
        code (str): Python code for visualization (should use matplotlib)
        stock_name (str): The stock name to use for the JPEG filename

    Returns:
        str: Path to the saved JPEG file or error message
    """
    logger.debug(f"Attempting to generate and save graph for stock: {stock_name}")
    # Prepare a local namespace for exec
    local_ns = {"plt": plt}
    try:
        logger.debug("Executing visualization code...")
        with contextlib.redirect_stdout(io.StringIO()):
            exec(code, {}, local_ns)
        
        jpeg_path = f"{stock_name}.jpeg"
        logger.debug(f"Code executed. Saving graph to {jpeg_path}")
        plt.savefig(jpeg_path, format="jpeg")
        plt.close()
        logger.debug(f"Graph successfully saved: {jpeg_path}")
        return f"Graph saved as {jpeg_path}"
    except Exception as e:
        logger.error(f"Error executing code or saving graph for {stock_name}: {e}", exc_info=True)
        plt.close()
        return f"Error executing code: {e}"
