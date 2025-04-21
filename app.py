from langchain_anthropic import ChatAnthropic
import os
from dotenv import load_dotenv

from typing import List
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

os.environ["LANGCHAIN_PROJECT"] = "German_Teacher_Tests.V0"
load_dotenv()
llm = ChatAnthropic(
    model="claude-3-sonnet-20240229",
    temperature=0
)


#Defining States
class State(TypedDict):
    state: str


#Defining Nodes
def initial_conversation(state):
    return {"state" : state["state"] + ", I am working"}


#'Nodes'
builder = StateGraph(State)
builder.add_node("initial_conversation", initial_conversation)

#Edges
#Edges alwas refer to the "naming" of the node, not the node itself*
builder.add_edge(START, "initial_conversation")
builder.add_edge("initial_conversation", END)
graph = builder.compile()


user_says = input("You:\n")
final_output = graph.invoke({"state" : f"{user_says}"})
print(final_output)