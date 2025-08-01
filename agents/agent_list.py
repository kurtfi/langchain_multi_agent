from agents.web_search_agent import create_web_search_agent
from agents.financial_agent import create_financial_agent
from agents.code_agent import create_code_agent

def create_financial_agent_with_prompt(llm, tools, system_prompt=None):
    return create_financial_agent(llm, tools, system_prompt)

def create_web_search_agent_with_prompt(llm, tools, system_prompt=None):
    return create_web_search_agent(llm, tools, system_prompt)

def create_code_agent_with_prompt(llm, tools, system_prompt=None):
    return create_code_agent(llm, tools, system_prompt)

agent_creators = {
    "FinancialAgent": create_financial_agent_with_prompt,
    "WebSearchAgent": create_web_search_agent_with_prompt,
    "CodeAgent": create_code_agent_with_prompt,
}
