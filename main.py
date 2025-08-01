import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from graph.builder import build_graph
from langgraph.graph import END
from dotenv import load_dotenv

# Load environment variables
from dotenv import load_dotenv
_ = load_dotenv()

def process_event(event):
    for key, value in event.items():
        if "messages" in value and value["messages"]:
            agent_name = value["messages"][-1].name
            content = value["messages"][-1].content
            print(f"--- {agent_name} ---\n{content}\n")
        if "next" in value:
            print(f"--- Supervisor ---\nSupervisor decides the next agent: {value['next']}\n")

def main():
    # Define the LLM
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    # Load the configuration
    with open("agent_config.json", 'r') as f:
        config_json = json.load(f)

    # Build the graph
    graph = build_graph(llm, config_json)
    thread_id = 1
    # Run the graph in a loop
    while True:
        thread_id += 1
        config = {"configurable": {"thread_id": str(thread_id)}}
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        events = graph.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config=config
        )
        for event in events:
            process_event(event)
            # Check if the process should finish
            if event.get("next") == "FINISH":
                print("Process finished.")
                return
            # Check for END node in the event keys
            if END in event:
                print("Process finished.")
                return

if __name__ == "__main__":
    main()
