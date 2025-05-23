from langchain_anthropic import ChatAnthropic
import os
from dotenv import load_dotenv

from typing_extensions import TypedDict

from langgraph.graph import START, END, StateGraph, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, AnyMessage


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
#Edges alwas refer to the "naming" of the node, not the node itself*
builder.add_edge(START, "agent_answers")
builder.add_edge("agent_answers", END)
graph = builder.compile()

#Better open Studio to wor on this, use   langgraph dev    in terminal

final_output = graph.invoke(
    {"messages": [HumanMessage(content="X")]},
    {"configurable": {"thread_id": "Test_1"}}  # or any unique identifier
)
