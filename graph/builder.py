import json
import functools
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from graph.state import AgentState
from agents.supervisor import create_supervisor_agent
from agents.agent_list import agent_creators
from tools.tool_registry import TOOL_REGISTRY
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

# Helper Function for Agent Nodes
def agent_node(state, agent, name, max_turns):
    # Initialize agent_turn_counts if not present
    agent_turn_counts = state.get("agent_turn_counts", {})
    agent_turn_counts = dict(agent_turn_counts)  # ensure mutable copy

    # Increment turn count for this agent
    agent_turn_counts[name] = agent_turn_counts.get(name, 0) + 1

    # Check if max_turns reached
    if agent_turn_counts[name] > max_turns:
        return {
            "messages": state["messages"],
            "next": "FINISH",
            "agent_turn_counts": agent_turn_counts
        }

    result = agent.invoke({"messages": state["messages"]})
    
    return {
        "messages": [AIMessage(content=result["messages"][-1].content, name=name)],
        "next": "Supervisor",  # Always route back to supervisor after agent execution
        "agent_turn_counts": agent_turn_counts
    }

def build_graph(llm: ChatOpenAI, config: dict):
    """
    Builds the multi-agent graph from a configuration dictionary.
    """
    agents = {}
    agent_nodes = {}

    # Get max_turns from config, default to 3 if not present
    max_turns = config.get("max_turns", 3)
    
    # Create agents and nodes
    for name, agent_config in config["agents"].items():
        creator = agent_creators.get(name)
        if creator:
            tool_names = agent_config.get("tools", [])
            tools = [TOOL_REGISTRY[t] for t in tool_names if t in TOOL_REGISTRY]
            system_prompt = agent_config.get("prompt")

            agent = creator(llm, tools, system_prompt)
            agents[name] = agent
            agent_nodes[name] = functools.partial(agent_node, agent=agent, name=name, max_turns=max_turns)

    # Supervisor
    supervisor_name = config["supervisor"]["name"]
    supervisor_prompt_text = config["supervisor"]["prompt"]
    supervisor_agent = create_supervisor_agent(
        llm, 
        {name: agent_config["description"] for name, agent_config in config["agents"].items()},
        supervisor_prompt_text,
        max_turns
    )
    
    # Initialize the graph
    workflow = StateGraph(AgentState)

    # Add nodes
    for name, node in agent_nodes.items():
        workflow.add_node(name, node)
    workflow.add_node(supervisor_name, supervisor_agent)

    # Define edges
    for relation in config["relations"]:
        if relation["edge_type"] == "edge":
            workflow.add_edge(relation["sender"], relation["receiver"])

    # Define conditional edges
    for conditional_edge in config["conditional_edges"]:
        sender = conditional_edge["sender"]
        receivers = conditional_edge["receivers"]
        conditional_map = {name: name for name in receivers.keys()}
        conditional_map["FINISH"] = END
        
        def route_next(state, conditional_map=conditional_map):
            return state["next"]
            
        workflow.add_conditional_edges(sender, route_next, conditional_map)

    # Entry point
    workflow.add_edge(START, supervisor_name)

    # Compile the graph with memory checkpointing
    memory = MemorySaver()
    graph = workflow.compile(checkpointer=memory)

    # Patch the initial state to include agent_turn_counts
    orig_stream = graph.stream
    def stream_with_turn_counts(input_state, *args, **kwargs):
        if "agent_turn_counts" not in input_state:
            input_state["agent_turn_counts"] = {}
        return orig_stream(input_state, *args, **kwargs)
    graph.stream = stream_with_turn_counts

    return graph
