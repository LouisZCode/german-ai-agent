from langchain_anthropic import ChatAnthropic
import os
from dotenv import load_dotenv

from langgraph.graph import START, END, StateGraph, MessagesState
from langchain_core.messages import HumanMessage

#For the database / Tools
from langgraph.prebuilt import ToolNode, tools_condition



os.environ["LANGCHAIN_PROJECT"] = "Conversation_Agent"
load_dotenv()
llm = ChatAnthropic(
    model="claude-3-sonnet-20240229",
    temperature=0
)


#TOOL
import os
import csv
import glob

import os
import csv
import glob

def retrieve_student_profile(db_path="./student_data"):
    """
    Read and display all student profiles from CSV files in the specified directory.
    
    Parameters:
    - db_path: Directory path where CSV files are stored (default: "./student_data")
    
    Returns:
    - student_count: Number of students found
    """
    # Check if directory exists
    if not os.path.exists(db_path):
        print(f"Directory not found: {db_path}")
        return 0
    
    # Get all CSV files in the directory
    csv_files = glob.glob(os.path.join(db_path, "*.csv"))
    
    if not csv_files:
        print(f"No student data files found in {db_path}")
        return 0
    
    student_count = 0
    
    # Process each file
    for file_path in csv_files:
        filename = os.path.basename(file_path)
        print(f"\n===== Student File: {filename} =====")
        
        try:
            with open(file_path, 'r', newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    student_count += 1
                    name = {row['name']}
                    language_level = {row['language_level']}
                    print(f"\nStudent ID: {row['id']}")
                    print(f"Name: {row['name']}")
                    print(f"Language Level: {row['language_level']}")
                    print(f"Registration Date: {row['registration_date']}")
                    
                    # Handle hobbies - convert from pipe-separated to list
                    hobbies = row['hobbies'].split('|') if row['hobbies'] else []
                    if hobbies:
                        print(f"Hobbies: {', '.join(hobbies)}")
                    else:
                        print("Hobbies: None")
        except Exception as e:
            print(f"Error reading file {filename}: {e}")
        
        return name, language_level, hobbies


llm_with_tools = llm.bind_tools([retrieve_student_profile])

#Defining Nodes
def conversation_agent(state: MessagesState):
    return {"messages" : [llm_with_tools.invoke(state["messages"])]}


#'Nodes'
builder = StateGraph(MessagesState)
builder.add_node("conversation_agent", conversation_agent)
builder.add_node("tools", ToolNode([retrieve_student_profile]))

#Edges
builder.add_edge(START, "conversation_agent")
builder.add_conditional_edges(
    "conversation_agent",
    tools_condition
)
builder.add_edge("tools", "conversation_agent")
builder.add_edge("conversation_agent", END)
graph = builder.compile()


final_output = graph.invoke(
    {"messages": [HumanMessage(content="Hi")]},
    {"configurable": {"thread_id": "conversation_agent_Test"}}  # or any unique identifier
)

