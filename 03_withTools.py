from langchain_anthropic import ChatAnthropic
import os
from dotenv import load_dotenv

from langgraph.graph import START, END, StateGraph, MessagesState
from langchain_core.messages import HumanMessage

#For the database / Tools
from langgraph.prebuilt import ToolNode, tools_condition
from tools import save_initial_profile


os.environ["LANGCHAIN_PROJECT"] = "Data_Gatherer_Prompt_Tools"
load_dotenv()
llm = ChatAnthropic(
    model="claude-3-sonnet-20240229",
    temperature=0
)


#TOOL
#import os
import csv
import datetime
import random
import string

def save_initial_profile(name, language_level, hobbies, student_id=None, db_path="./student_data"):
    """
    Save a student profile to a CSV file with filename format: dateofcreation_name_ID.csv
    
    Parameters:
    - name: Student's name (string)
    - language_level: Student's language level (string, e.g., "Beginner A1")
    - hobbies: List of up to 3 hobbies (list of strings)
    - student_id: Pre-existing student ID (optional, will generate random alphanumeric if None)
    - db_path: Base directory path where to save the CSV files (default: "./student_data")
    
    Returns:
    - student_id: ID of the student
    - filename: Path to the saved CSV file
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
        
        # Write header
        writer.writerow(['id', 'name', 'language_level', 'registration_date', 'hobbies'])
        
        # Write student data
        writer.writerow([student_id, name, language_level, current_time, hobby_str])
    
    return student_id, full_path

llm_with_tools = llm.bind_tools([save_initial_profile])

#Defining Nodes
def gathering_agent(state: MessagesState):
    return {"messages" : [llm_with_tools.invoke(state["messages"])]}

def farewell_node(state: MessagesState):
    # Add a system message to guide the LLM to provide a farewell
    farewell_message = HumanMessage(content="The student profile has been created successfully. Please provide a friendly farewell message to the user, summarizing what was done and ending the conversation.")
    # Add this instruction to the existing messages
    updated_messages = state["messages"] + [farewell_message]
    # Invoke the LLM with the updated messages
    return {"messages": state["messages"] + [llm.invoke(updated_messages)]}


#'Nodes'
builder = StateGraph(MessagesState)
builder.add_node("gathering_agent", gathering_agent)
builder.add_node("tools", ToolNode([save_initial_profile]))
builder.add_node("farewell_node", farewell_node)

#Edges
builder.add_edge(START, "gathering_agent")
builder.add_conditional_edges(
    "gathering_agent", 
    tools_condition,
    )  #This should direct to the tools in case it is needed.
builder.add_edge("tools", "farewell_node")
builder.add_edge("farewell_node", END)
graph = builder.compile()


final_output = graph.invoke(
    {"messages": [HumanMessage(content="Hi")]},
    {"configurable": {"thread_id": "Test_2"}}  # or any unique identifier
)