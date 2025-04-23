from langchain_anthropic import ChatAnthropic
import os
from dotenv import load_dotenv

from typing_extensions import TypedDict

from langgraph.graph import START, END, StateGraph, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, AnyMessage

#For the database / Tools
from langgraph.prebuilt import ToolNode, tools_condition
from tools import save_initial_profile

from system_prompts import info_taker_Agent_sys_prompt


os.environ["LANGCHAIN_PROJECT"] = "German_Teacher_Chatbot_V1"
load_dotenv()
llm = ChatAnthropic(
    model="claude-3-sonnet-20240229",
    temperature=0
)

#NOTE TOOL ADD to agent
llm_with_tools = llm.bind_tools([save_initial_profile])

#Memory Class
memory = MemorySaver()


#Defining States
class MessageState(TypedDict):
    pass


#Defining Nodes
def gathering_agent(state: MessagesState):
    messages = state["messages"]
    llm_answer = llm_with_tools.invoke(messages)
    # Return the changes to be applied to the state
    return {"messages": messages + [AIMessage(content=llm_answer.content)]}



#'Nodes'
builder = StateGraph(MessagesState)
builder.add_node("gathering_agent", gathering_agent)
builder.add_node("tools", ToolNode([save_initial_profile]))

#Edges
builder.add_edge(START, "gathering_agent")
builder.add_conditional_edges(
    "gathering_agent", 
    tools_condition)  #This should direct to the tools in case it is needed.
builder.add_edge("tools", "gathering_agent")
graph = builder.compile(checkpointer=memory)



#InitialAnswer
first_answer = graph.invoke(
    {"messages": [
        SystemMessage(content=info_taker_Agent_sys_prompt), 
        HumanMessage(content="Start the conversation.")
    ]},
    {"configurable": {"thread_id": "Test_1"}}
)

# Display AI's first message (greeting/introduction)
ai_message = first_answer["messages"][-1]
print(f"\nAI:\n{ai_message.content}")



#Start the conversation
chatting = True
while chatting:

    user_input = input("\nYou:\n")

    
    if user_input == "bye":
        chatting = False

    else:
        final_output = graph.invoke(
    {"messages": first_answer["messages"] + [HumanMessage(content=user_input)]},
    {"configurable": {"thread_id": "Test_1"}}  # or any unique identifier
)
        first_answer = final_output

        # Extract and print just the AI's response
        ai_message = final_output["messages"][-1]  
        print(f"\nAI:\n{ai_message.content}")