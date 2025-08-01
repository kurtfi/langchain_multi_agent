from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from tools.alpha_vantage import alpha_vantage_tool
from tools.date import get_current_date

def create_financial_agent(llm: ChatOpenAI, tools, system_prompt: str = None):
    """
    Creates a financial analysis agent.
    """
    tool_instances = []
    for tool in tools:
        if isinstance(tool, str):
            from tools.tool_registry import TOOL_REGISTRY
            tool_instances.append(TOOL_REGISTRY[tool])
        else:
            tool_instances.append(tool)

    if system_prompt is None:
        system_prompt = "You are a financial analysis agent. Your role is to use the Alpha Vantage tool to gather financial data and provide concise, informative answers. " \
                       "When a user asks about a stock, always use the Alpha Vantage tool to retrieve the latest data. " \
                       "Do not generate charts or plots. Only use the tools provided to you and return a clear, text-based analysis or result."
    financial_agent = create_react_agent(llm, tools=tool_instances, state_modifier=system_prompt)
    return financial_agent
