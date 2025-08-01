from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent


def create_code_agent(llm: ChatOpenAI, tools, system_prompt: str = None):
    """
    Creates a code agent that generates visualizations and can save graphs as JPEGs.

    Args:
        llm (ChatOpenAI): The language model to use for the agent
        tools (list): List of tool names (str) or tool instances
        system_prompt (str): Optional system prompt for the agent

    Returns:
        The configured agent for code visualization
    """
    tool_instances = []
    for tool in tools:
        if isinstance(tool, str):
            from tools.tool_registry import TOOL_REGISTRY
            tool_instances.append(TOOL_REGISTRY[tool])
        else:
            tool_instances.append(tool)

    if system_prompt is None:
        system_prompt = (
            "You are a visualization agent. Your role is to create visual representations of data using Python. "
            "Use the available tools to submit your visualization code or to execute and save graphs as JPEGs. "
            "Do not perform any data analysis or gather information. Your sole purpose is to take the given data "
            "and create appropriate visualizations. Return the code for the visualization for review, or save the graph as instructed."
        )
    code_agent = create_react_agent(llm, tools=tool_instances, state_modifier=system_prompt)
    return code_agent
