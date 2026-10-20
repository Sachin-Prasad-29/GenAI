from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import  add_messages
from langgraph.checkpoint.memory import InMemorySaver
from typing import Annotated
import streamlit as st

load_dotenv()


class ChatState(BaseModel):
    messages: Annotated[list,add_messages]

llm = ChatGroq(model="openai/gpt-oss-20b")


def chat_bot_node(state:ChatState) -> ChatState:
    res = llm.invoke(state.messages)
    state.messages = [res]
    return state

memory = InMemorySaver()


graph = StateGraph(ChatState)
graph.add_node("chatBot",chat_bot_node)
graph.add_edge(START,"chatBot")
graph.add_edge("chatBot",END)
graph = graph.compile(checkpointer=memory)


config = {"configurable":{"thread_id":"123"}}

# Testing the Graph
while True:
    query = input("User : ")
    if query.lower() in ['exit','quit']:
        print("Good Bye ")
        break

    messages = [{"role":"user","content":query}]
    response = graph.invoke({"messages":messages},config)

    print("AI : " + response["messages"][-1].content)
    

