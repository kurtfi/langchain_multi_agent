from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field, create_model
from typing import Literal, Dict, List

def create_route_response_model(members: List[str]):
    """
    Dynamically creates the RouteResponse model with a Literal type 
    for the 'next' field, based on the provided list of members.
    """
    # Add "FINISH" to the list of possible next steps
    possible_next_steps = Literal[tuple(["FINISH"] + members)]
    
    RouteResponse = create_model(
        'RouteResponse',
        next=(possible_next_steps, Field(
            ..., 
            description="The next agent to act or 'FINISH' to end the conversation."
        )),
        __base__=BaseModel,
        __doc__="The supervisor's response to the user's request."
    )
    
    return RouteResponse

def create_supervisor_agent(llm: ChatOpenAI, members: dict, supervisor_prompt_text: str, max_turns: int = 3):
    """
    Creates the supervisor agent.
    """
    # Create the RouteResponse model dynamically
    RouteResponse = create_route_response_model(list(members.keys()))
    
    # Use the provided prompt text and concatenate agent descriptions
    agent_descriptions = "\n".join([f"- {name}: {description}" for name, description in members.items()])
    full_prompt = f"{supervisor_prompt_text} Available agents and their descriptions: {agent_descriptions}"
    
    # Supervisor Prompt
    supervisor_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", full_prompt),
            MessagesPlaceholder(variable_name="messages"),
            (
                "system",
                "Based on the conversation, who should act next? Choose one of: {options}",
            ),
        ]
    ).partial(options=str(["FINISH"] + list(members.keys())))

    # Supervisor Agent Function
    supervisor_chain = supervisor_prompt | llm.with_structured_output(RouteResponse)
    
    # Wrap the supervisor chain to filter out agents that have exceeded max_turns
    def supervisor_with_turn_limit(state):
        # Get agent turn counts from state
        agent_turn_counts = state.get("agent_turn_counts", {})
        
        # Filter out agents that have exceeded max_turns
        available_agents = [
            agent for agent in members.keys() 
            if agent_turn_counts.get(agent, 0) < max_turns
        ]
        
        # If no agents are available, force FINISH
        if not available_agents:
            return {"next": "FINISH"}
        
        # If agents are available, use the supervisor chain
        result = supervisor_chain.invoke(state)
        return {"next": result.next}
    
    return supervisor_with_turn_limit
