from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader,PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
import streamlit as st

# data in st session
if "document_uploaded" not in st.session_state:
    st.session_state["document_uploaded"] = False

if "agent"  not in st.session_state:
    st.session_state["agent"] = None

if "vector_store" not in st.session_state:
    st.session_state["vector_store"] = None

if "messages" not in st.session_state:
    st.session_state["messages"] = [] 


def process_document(path):

    # Load the Documents
    loader = PyPDFDirectoryLoader(path)
    docs = loader.load()


    # Split the Docs in chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    docs = splitter.split_documents(documents=docs)

    # Embedding and vector DB
    embeddings = NVIDIAEmbeddings(model="nvidia/nemotron-3-embed-1b")
    vector_store = InMemoryVectorStore.from_documents(
        documents=docs,
        embedding=embeddings
    )

    # Create Agent 
    # llm
    llm = ChatGroq(model="openai/gpt-oss-20b")

    # tool
    @tool
    def retrieve_context(query:str):
        """
            Retrieve the most relevant context from the vector store based on the query.
        """
        context = ""
        docs = vector_store.similarity_search(query=query,k=3)
        for doc in docs:
            context += doc.page_content + "\n\n"
        return context

    # System Prompt
    system_prompt="""
        You are a helpful assistant that answers questions based on the context provided.
        Use the following tools to answer the question.
        tools : ['retrieve_context']
        Also If the question is out of the context you should answer that the question is out of the context.
        Context: {context}
        Question: {question}
        Answer:
    """


    # create Agent
    memory = InMemorySaver()

    agent = create_agent(
        model=llm,
        tools=[retrieve_context],
        system_prompt=system_prompt,
        checkpointer=memory
    )

    st.session_state.agent = agent
    st.session_state.document_uploaded = True


# Upload UI 
if not st.session_state.document_uploaded:
    st.subheader("Upload You Documents....")
    uploaded = st.file_uploader(label="Upload your PDF",type=["pdf"],accept_multiple_files=True)
    if uploaded:
        with st.spinner("Processing...."):
            path = "./doc_files/"
            for file in uploaded:
                with open(path + file.name, "wb") as f:
                    f.write(file.getvalue())

            process_document(path)
            st.rerun()

# Chat UI

if st.session_state.document_uploaded and st.session_state.agent:
    st.subheader("Ask Questions about the Documents....")
    
    for message in st.session_state.messages:
        role = message.get("role")
        content = message.get("content")
        st.chat_message(role).markdown(content)
        
    query = st.chat_input("Ask anything related to uploaded documents....")

    if query:
        st.session_state.messages.append({"role":"user","content":query})
        st.chat_message("user").markdown(query)
        response = st.session_state.agent.invoke(
            {"messages":[{"role":"user", "content":query}]},
            {"configurable":{"thread_id":"doc_1"}}
        )
        answer = response["messages"][-1].content
        st.chat_message("ai").markdown(answer)
        st.session_state.messages.append({"role":"ai","content":answer})