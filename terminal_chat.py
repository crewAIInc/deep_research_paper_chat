from src.deep_research_agent.main import DeepResearchFlow
from colorama import Fore, Style
import uuid

# First chat message
print(f"{Fore.BLUE}{Style.BRIGHT}This is a terminal chat with Lead Generation crew.{Style.RESET_ALL} (type 'exit' to quit)")
id = None

while True:
    user_input = input("> ")
    if user_input == "exit":
        break

    # Create flow instance
    chat_flow = DeepResearchFlow()
    
    # Prepare inputs - create new ID if this is first interaction
    inputs = {
        "user_message": user_input
    }
    
    # If we have an existing conversation ID, use it to maintain state
    if id is not None:
        inputs["id"] = id

    # Execute the flow
    response = chat_flow.kickoff(inputs=inputs)

    # Display response - check if it's a structured MessageOutput or raw text
    if hasattr(response, 'content'):
        print(f"{Fore.GREEN}{Style.BRIGHT}Assistant:{Style.RESET_ALL} {response.content}")
    else:
        print(f"{Fore.GREEN}{Style.BRIGHT}Assistant:{Style.RESET_ALL} {response}")
    
    # Display only the latest user message (dimmed for context)
    if hasattr(chat_flow.state, 'message_history') and chat_flow.state.message_history:
        # Get the latest user and assistant messages, filtering out None values
        latest_messages = [msg for msg in chat_flow.state.message_history[-2:] if msg is not None] if len(chat_flow.state.message_history) >= 2 else [msg for msg in chat_flow.state.message_history if msg is not None]
        if latest_messages:
            print(f"\n{Style.DIM}--- Latest Exchange ---")
            for msg in latest_messages:
                role_color = Fore.CYAN if msg.role == "user" else Fore.YELLOW
                print(f"{role_color}{msg.role.capitalize()}:{Style.RESET_ALL} {msg.content}")
            print(f"--- End Latest ---{Style.RESET_ALL}")
    
    print()  # Add blank line for readability

    # Store the ID for next iteration to maintain conversation context
    if hasattr(chat_flow.state, 'id'):
        id = chat_flow.state.id
    elif id is None:
        # Generate a new ID if one wasn't created
        id = str(uuid.uuid4())