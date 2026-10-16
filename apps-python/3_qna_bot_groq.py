"""
Requirments : 
    LLM
    Tools = Google Search Tool
    Agent
    Memory
    Streaming
    web interface
"""

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
import streamlit as st

load_dotenv()

llm = ChatGroq(model='openai/gpt-oss-20b',streaming=True)
search = GoogleSerperAPIWrapper()

tools = [search.run]

if 'memory' not in st.session_state:
    st.session_state.memory = MemorySaver()
    st.session_state.history = []

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="You are a QnA Bot and can search for any Question in the Google Search Engine",
    checkpointer=st.session_state.memory
)


# Building WEB InterFace

st.subheader("QnA BOT")
for messsage in st.session_state.history:
    role = messsage['role']
    content = messsage['content']
    st.chat_message(role).markdown(content)
    
query = st.chat_input("Ask Anything ?")
if query:
    st.chat_message("user").markdown(query)
    st.session_state.history.append({'role':'user','content':query})

    response = agent.stream({'messages':[{'role':'user','content':query}]},{'configurable': {'thread_id': "sac123"}},stream_mode="messages")

    # answer = response['messages'][-1].content
    # st.chat_message("ai").markdown(answer)

    ai_container = st.chat_message("ai")
    with ai_container:
        space = st.empty()

        message = ""

        for chunk in response:
            message = message + chunk[0].content
            space.write(message)

        st.session_state.history.append({'role':'ai','content':message})
 


