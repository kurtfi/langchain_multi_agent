from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import create_react_agent
from tools.date import get_current_date

def create_web_search_agent(llm: ChatOpenAI, tools, system_prompt: str = None):
    """
    Creates a web search agent.
    """
    tool_instances = []
    for tool in tools:
        if isinstance(tool, str):
            from tools.tool_registry import TOOL_REGISTRY
            tool_instances.append(TOOL_REGISTRY[tool])
        else:
            tool_instances.append(tool)

    if system_prompt is None:
        system_prompt = "You are a web search agent. Your role is to use web search tools to find information and return comprehensive answers to queries."
    web_search_agent = create_react_agent(llm, tools=tool_instances, state_modifier=system_prompt)
    return web_search_agent
