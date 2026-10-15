from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
import streamlit as st

out = StrOutputParser()

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

st.title("First CHATBOT 🤖")
st.markdown('My Bot with Langchain and Google GEMINI 🔥🔥')


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])


query  = st.chat_input("Ask anything ...")

if query:
    st.session_state.messages.append({'role':'user','content':query})
    st.chat_message('user').markdown(query)
    result = out.invoke(llm.invoke(query))
    st.session_state.messages.append({'role':'ai','content':result})
    st.chat_message('ai').markdown(result)