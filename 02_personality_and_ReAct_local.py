from langchain_anthropic import ChatAnthropic
import os
from dotenv import load_dotenv

from typing_extensions import TypedDict

from langgraph.graph import START, END, StateGraph, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, AnyMessage


from system_prompts import info_taker_Agent_sys_prompt


os.environ["LANGCHAIN_PROJECT"] = "German_Teacher_Chatbot_V1"
load_dotenv()
llm = ChatAnthropic(
    model="claude-3-sonnet-20240229",
    temperature=0
)



#Memory Class
memory = MemorySaver()

#Defining States
class MessageState(TypedDict):
    pass

#Defining Nodes
def agent_answers(state: MessagesState):
    messages = state["messages"]
    llm_answer = llm.invoke(messages)
    # Return the changes to be applied to the state
    return {"messages": messages + [AIMessage(content=llm_answer.content)]}


#'Nodes'
builder = StateGraph(MessagesState)
builder.add_node("agent_answers", agent_answers)

#Edges
builder.add_edge(START, "agent_answers")
builder.add_edge("agent_answers", END)
graph = builder.compile(checkpointer=memory)

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

#Start the ocnversation
chatting = True
while chatting:

    user_input = input("\nYou:\n")

    
    if user_input == "bye":
        chatting = False

    else:
        final_output = graph.invoke(
    {"messages": [HumanMessage(content=user_input)]},
    {"configurable": {"thread_id": "Test_1"}}  # or any unique identifier
)
        # Extract and print just the AI's response
        ai_message = final_output["messages"][-1]  
        print(f"\nAI:\n{ai_message.content}")