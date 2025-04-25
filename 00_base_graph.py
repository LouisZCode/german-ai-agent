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
def test_node(state):
    #print("---TEST NODE---")
    return {"state" : state["state"] + " the test was successful!"}


#'Nodes'
builder = StateGraph(State)
builder.add_node("test_node", test_node)

#Edges
#Edges alwas refer to the "naming" of the node, not the node itself*
builder.add_edge(START, "test_node")
builder.add_edge("test_node", END)
graph = builder.compile()

#We invoke the graph with an initial state
final_output = graph.invoke({"state" : "Running Test! and..."})
print(final_output)