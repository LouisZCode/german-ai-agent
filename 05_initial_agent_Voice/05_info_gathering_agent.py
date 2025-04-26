from langchain_anthropic import ChatAnthropic
import os
import csv
import datetime
import random
import string
from dotenv import load_dotenv
import json

from langgraph.graph import START, END, StateGraph, MessagesState
from langchain_core.messages import HumanMessage, SystemMessage

from langgraph.checkpoint.memory import MemorySaver
from agent_stt_module import SpeechToText

# Initialize components
stt = SpeechToText()
os.environ["LANGCHAIN_PROJECT"] = "info_gathering_voice"
load_dotenv()
llm = ChatAnthropic(
    model="claude-3-5-haiku-20241022",
    temperature=0
)
memory = MemorySaver()

sys_prompt = """You are an AI Assistant designed to gather information about an user. You do not have the capacity nor will help in any other way.
You are concrete and use only the most needed words to ask the questions to get the information.

You start the conversation by greeting the student with a "Hi and welcome your AI powered German course!, before we begin, I need to gather some information about yourself
so we can personalize your learning experience"

Your goal is to fill out this format:
Student_name: {name}, students_german_level: {level}, students_interests: {list_of_interests}, Student_ID: {5_random_characters}

The level should be categorized between beginner, intermediate and advanced. You ask if they are beginner, intermediate or advanced.
If the student answers:
A1 or A2 = beginner.
B1 or B2 = intermediate.
C1 or C2 = advanced.
There is no other format that is accepted. Only beginner, intermediate or advanced are needed to be gathered.
In case the student is not sure, you can ask about the experiences of the student, and propose a level. This level can only be saved if the student agrees to it.

The hobbies need to be at least 3. Less than 3 is not acceptable. However, if the student gives more than 3, you are allowed to save as many as 10. If the student is having difficulties coming up with 3 hobbies, this, you can propose common hobbies as options (for example, traveling, going to restaurants, meeting with friends..).
Do not propose hobbies until the student has difficulties coming up with them.
Hobbies cannot be too personal, for example, you do not accept sex, drug or harmful related topics.

You have access to the following tool:
Tool Name: save_initial_profile, Description: Creates a unique file with the base data of the student, Arguments: name: str, level: str, Hobbies: list

Only when you have all the information at hand, call the tool so we can save that information in our database.
"""

def save_initial_profile(name, language_level, hobbies, student_id=None, db_path="./05_initial_agent_Voice/student_data"):
    """
    Save a student profile to a CSV file with filename format: dateofcreation_name_ID.csv
    """
    # Create directory if needed
    os.makedirs(db_path, exist_ok=True)
    
    # Get current timestamp for both the filename and data
    current_date = datetime.datetime.now().strftime("%Y%m%d")
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Generate student_id if not provided
    if student_id is None:
        # Generate a random 8-character alphanumeric ID
        chars = string.ascii_uppercase + string.digits
        student_id = ''.join(random.choice(chars) for _ in range(8))
        
        # Ensure the ID is unique by checking existing files
        existing_ids = set()
        if os.path.exists(db_path):
            for filename in os.listdir(db_path):
                if filename.endswith('.csv'):
                    try:
                        file_id = filename.split('_')[-1].split('.')[0]
                        existing_ids.add(file_id)
                    except (ValueError, IndexError):
                        pass
        
        # Regenerate if there's a collision (unlikely but possible)
        while student_id in existing_ids:
            student_id = ''.join(random.choice(chars) for _ in range(8))
    
    # Create filename using the required format: dateofcreation_name_ID.csv
    # Replace spaces in name with underscores for the filename
    safe_name = name.replace(' ', '_')
    filename = f"{current_date}_{safe_name}_{student_id}.csv"
    full_path = os.path.join(db_path, filename)
    
    # Format hobbies as a pipe-separated string (up to 3)
    hobby_str = "|".join([h.strip() for h in hobbies[:3] if h.strip()])
    
    # Write to CSV file
    with open(full_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['id', 'name', 'language_level', 'registration_date', 'hobbies'])
        writer.writerow([student_id, name, language_level, current_time, hobby_str])
    
    return student_id, full_path

def listen_and_gathering_agent(state: MessagesState):
    print("\n🎤 Listening... (speak to start)")
    
    # Record and transcribe using VAD
    transcribed_text = stt.capture_and_transcribe()
    
    if not transcribed_text:
        transcribed_text = "I couldn't understand what you said. Could you please repeat?"
    
    print(f"🔊 You:\n \"{transcribed_text}\"")
    
    # Get existing messages and add user message
    messages = state.get("messages", [])
    user_message = {"role": "user", "content": transcribed_text}
    messages.append(user_message)
    
    # Check if there's a pending tool call in previous messages
    contains_tool_call = False
    for msg in messages:
        content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "tool_use":
                    contains_tool_call = True
                    tool_id = item.get("id")
                    tool_name = item.get("name")
                    tool_args = item.get("input", {})
                    
                    # Execute the tool
                    print(f"Executing tool: {tool_name} with args: {tool_args}")
                    result = save_initial_profile(**tool_args)
                    print(f"Profile saved! ID: {result[0]}, File: {result[1]}")
                    
                    # Generate farewell separately
                    farewell_prompt = f"""
                    The student profile for Luis has been successfully created with ID {result[0]}. 
                    The profile includes their German level (beginner) and hobbies.
                    Please provide a friendly farewell message to the user, thanking them and wishing them
                    well on their German learning journey.
                    """
                    
                    farewell = llm.invoke(farewell_prompt)
                    print(f"\n🤖 Claude (Farewell):\n \"{farewell.content}\"")
                    
                    # Save state and end conversation
                    save_conversation(messages)
                    return {"messages": messages, "next": "__end__"}
    
    # If no pending tool call, get next AI response
    ai_response = llm_with_tools.invoke(messages)
    
    # Format for display
    if isinstance(ai_response.content, list):
        text_parts = []
        for item in ai_response.content:
            if isinstance(item, dict) and item.get('type') == 'text':
                text_parts.append(item.get('text', ''))
        display_text = '\n'.join(text_parts)
    else:
        display_text = ai_response.content
        
    print(f"\n🤖 Claude:\n \"{display_text}\"")
    
    # Check if this new response contains a tool call
    if isinstance(ai_response.content, list):
        for item in ai_response.content:
            if isinstance(item, dict) and item.get("type") == "tool_use":
                tool_name = item.get("name")
                tool_args = item.get("input", {})
                
                print(f"Executing tool: {tool_name} with args: {tool_args}")
                result = save_initial_profile(**tool_args)
                print(f"Profile saved! ID: {result[0]}, File: {result[1]}")
                
                # Generate farewell separately
                farewell_prompt = f"""
                Tell the student that his or her profile has been successfully created with ID {result[0]}. 
                You have to say goodbye to the student and wish him or her luck in his or her German adventure in a fun and polite way!
                Directly start this message, do not tell me "here is the message" before.
                """
                
                farewell = llm.invoke(farewell_prompt)
                print(f"\n🤖 Claude (Farewell):\n \"{farewell.content}\"")
                
                # Add AI response to messages and save
                messages.append(ai_response)
                save_conversation(messages)
                
                # End the conversation
                return {"messages": messages, "next": "__end__"}
    
    # If no tool call, continue normally
    messages.append(ai_response)
    save_conversation(messages)
    return {"messages": messages}

# Set up LLM with tools
llm_with_tools = llm.bind_tools([save_initial_profile])

# Build the graph
builder = StateGraph(MessagesState)
builder.add_node("listen_and_gathering_agent", listen_and_gathering_agent)
builder.add_edge(START, "listen_and_gathering_agent")
builder.add_edge("listen_and_gathering_agent", END)
graph = builder.compile(checkpointer=memory)

# File management functions
def save_conversation(messages, file_path="./05_initial_agent_Voice/conversation_state.json"):
    """Save the current conversation messages to a JSON file with pretty formatting"""
    serializable_messages = []
    for msg in messages:
        if hasattr(msg, 'model_dump'):
            msg_dict = msg.model_dump()
        elif hasattr(msg, 'dict'):
            msg_dict = msg.dict()
        elif isinstance(msg, dict):
            msg_dict = msg.copy()
            
            if 'role' in msg_dict and msg_dict['role'] == 'user':
                msg_dict['_message_type'] = 'human'
            elif 'role' in msg_dict and msg_dict['role'] == 'assistant':
                msg_dict['_message_type'] = 'ai'
        
        if hasattr(msg, 'type') and msg.type:
            msg_dict['_message_type'] = msg.type
            
        serializable_messages.append(msg_dict)
    
    with open(file_path, 'w') as f:
        json.dump(serializable_messages, f, indent=2)

def load_conversation(file_path="./05_initial_agent_Voice/conversation_state.json"):
    """Load conversation messages from a JSON file"""
    if not os.path.exists(file_path):
        return None
    
    with open(file_path, 'r') as f:
        serialized_messages = json.load(f)
    
    messages = []
    for msg in serialized_messages:
        msg_type = msg.get('_message_type') or msg.get('type')
        
        if not msg_type:
            if 'role' in msg and msg['role'] == 'user':
                msg_type = 'human'
            elif 'role' in msg and msg['role'] == 'assistant':
                msg_type = 'ai'
        
        if msg_type == 'system':
            messages.append(SystemMessage(content=msg.get('content', '')))
        elif msg_type == 'human':
            messages.append(HumanMessage(content=msg.get('content', '')))
        elif msg_type == 'ai':
            if isinstance(msg.get('content'), list):
                ai_message = {"role": "assistant", "content": msg.get('content')}
            else:
                ai_message = {"role": "assistant", "content": msg.get('content', '')}
            messages.append(ai_message)
        elif 'role' in msg and msg['role'] == 'user':
            messages.append({"role": "user", "content": msg.get('content', '')})
        elif 'role' in msg and msg['role'] == 'assistant':
            messages.append({"role": "assistant", "content": msg.get('content', '')})
    
    return messages

# Main execution
initial_messages = load_conversation()

if initial_messages:
    final_output = graph.invoke(
        {"messages": initial_messages},
        {"configurable": {"thread_id": "voice_memory_test"}}
    )
else:
    final_output = graph.invoke(
        {"messages": [SystemMessage(content=sys_prompt)]},
        {"configurable": {"thread_id": "voice_memory_test"}}
    )