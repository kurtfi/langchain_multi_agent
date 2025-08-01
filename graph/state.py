from typing import Annotated, Sequence
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage
import operator

class AgentState(TypedDict):
    messages: Annotated[Sequence[HumanMessage], operator.add]
    next: str
    agent_turn_counts: dict[str, int]
