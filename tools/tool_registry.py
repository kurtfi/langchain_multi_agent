from tools.alpha_vantage import alpha_vantage_tool
from tools.date import get_current_date
from tools.save_graph_tool import execute_and_save_graph_tool
from langchain_tavily import TavilySearch
from langchain_experimental.tools import PythonREPLTool


# Instantiate tools that require setup
tavily_tool = TavilySearch(max_results=5)
python_repl_tool = PythonREPLTool()

# Map tool names to actual tool instances
TOOL_REGISTRY = {
    "alpha_vantage": alpha_vantage_tool,
    "get_current_date": get_current_date,
    "tavily_search": tavily_tool,
    "python_repl": python_repl_tool,
    "execute_and_save_graph_tool": execute_and_save_graph_tool
}
