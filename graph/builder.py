"""
This module provides functions to dynamically build a multi-agent graph 
using LangGraph based on a provided configuration file.

The main components are:
- `agent_node`: A function that wraps an agent's execution, managing its turn count
  to prevent infinite loops.
- `build_graph`: The core function that constructs the entire agentic graph,
  including agents, a supervisor, and the connections between them, as defined
  in a configuration dictionary.
"""
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

def agent_node(state: AgentState, agent, name: str, max_turns: int) -> dict:
    """
    A node in the graph that executes a specific agent.

    This function wraps the agent's invocation. It tracks the number of times
    each agent has been called within a single run. If an agent exceeds a
    predefined `max_turns` limit, it forces the graph to finish, preventing
    infinite loops or excessive processing.

    Args:
        state (AgentState): The current state of the graph, containing messages
                            and other shared information.
        agent: The agent instance to be executed.
        name (str): The name of the agent being executed.
        max_turns (int): The maximum number of times this agent is allowed to
                         be called in a single graph execution.

    Returns:
        dict: A dictionary containing the agent's output message, the next
              node to transition to (usually the Supervisor), and the updated
              turn counts.
    """
    # Ensure agent_turn_counts is a mutable copy from the state
    agent_turn_counts = dict(state.get("agent_turn_counts", {}))

    # Increment the turn count for the current agent
    agent_turn_counts[name] = agent_turn_counts.get(name, 0) + 1

    # Check if the agent has exceeded its maximum number of turns
    if agent_turn_counts[name] > max_turns:
        # If so, force a finish by routing to the special "FINISH" end node
        return {
            "messages": state["messages"],
            "next": "FINISH",
            "agent_turn_counts": agent_turn_counts
        }

    # Invoke the agent with the current state's messages
    result = agent.invoke({"messages": state["messages"]})
    
    # Return the result, routing back to the supervisor for the next decision
    return {
        "messages": [AIMessage(content=result["messages"][-1].content, name=name)],
        "next": "Supervisor",
        "agent_turn_counts": agent_turn_counts
    }

def build_graph(llm: ChatOpenAI, config: dict) -> StateGraph:
    """
    Builds and compiles the multi-agent graph from a configuration dictionary.

    This function orchestrates the creation of the entire agentic system.
    It reads the configuration to:
    1. Instantiate each agent with its specific tools and system prompt.
    2. Create a supervisor agent responsible for routing tasks.
    3. Construct a `StateGraph` and add all agents and the supervisor as nodes.
    4. Define the edges (transitions) between nodes as specified in the config.
    5. Set up conditional routing based on the supervisor's decisions.
    6. Compile the graph, enabling memory for state persistence.

    Args:
        llm (ChatOpenAI): The language model instance to be used by all agents.
        config (dict): A dictionary containing the entire graph configuration,
                       including agent definitions, supervisor settings, and
                       relationships (edges).

    Returns:
        StateGraph: The compiled, executable LangGraph instance.
    """
    agents = {}
    agent_nodes = {}

    # Get max_turns from config, with a default value if not specified
    max_turns = config.get("max_turns", 3)
    
    # Create each agent and its corresponding graph node from the config
    for name, agent_config in config["agents"].items():
        creator = agent_creators.get(name)
        if creator:
            # Assemble the tools for this agent from the central tool registry
            tool_names = agent_config.get("tools", [])
            tools = [TOOL_REGISTRY[t] for t in tool_names if t in TOOL_REGISTRY]
            system_prompt = agent_config.get("prompt")

            # Create the agent instance
            agent = creator(llm, tools, system_prompt)
            agents[name] = agent
            # Create a partial function for the agent node to fix its arguments
            agent_nodes[name] = functools.partial(agent_node, agent=agent, name=name, max_turns=max_turns)

    # Create the supervisor agent
    supervisor_name = config["supervisor"]["name"]
    supervisor_prompt_text = config["supervisor"]["prompt"]
    supervisor_agent = create_supervisor_agent(
        llm, 
        # Provide the supervisor with descriptions of all agents it can route to
        {name: agent_config["description"] for name, agent_config in config["agents"].items()},
        supervisor_prompt_text,
        max_turns
    )
    
    # Initialize the graph with the defined state structure
    workflow = StateGraph(AgentState)

    # Add all agent nodes and the supervisor node to the graph
    for name, node in agent_nodes.items():
        workflow.add_node(name, node)
    workflow.add_node(supervisor_name, supervisor_agent)

    # Define direct edges (unconditional transitions) between nodes
    for relation in config["relations"]:
        if relation["edge_type"] == "edge":
            workflow.add_edge(relation["sender"], relation["receiver"])

    # Define conditional edges, which are used for routing from the supervisor
    for conditional_edge in config["conditional_edges"]:
        sender = conditional_edge["sender"]
        receivers = conditional_edge["receivers"]
        # Create a mapping from the 'next' state value to the actual node name
        conditional_map = {name: name for name in receivers.keys()}
        conditional_map["FINISH"] = END  # Map "FINISH" to the graph's end
        
        # The routing function simply returns the 'next' value from the state
        def route_next(state, conditional_map=conditional_map):
            return state["next"]
            
        workflow.add_conditional_edges(sender, route_next, conditional_map)

    # Define the entry point of the graph
    workflow.add_edge(START, supervisor_name)

    # Compile the graph, enabling memory for persisting state across runs
    memory = MemorySaver()
    graph = workflow.compile(checkpointer=memory)

    # Patch the graph's stream method to initialize agent_turn_counts in the input state
    # This ensures the turn counting mechanism works correctly from the start.
    orig_stream = graph.stream
    def stream_with_turn_counts(input_state, *args, **kwargs):
        if "agent_turn_counts" not in input_state:
            input_state["agent_turn_counts"] = {}
        return orig_stream(input_state, *args, **kwargs)
    graph.stream = stream_with_turn_counts

    return graph
