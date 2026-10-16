from dotenv import load_dotenv
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()
llm = ChatGroq(model='openai/gpt-oss-20b')
search = GoogleSerperAPIWrapper()

agent = create_agent(
    model=llm,
    tools=[search.run],
    system_prompt="You are a Agent and can Search for any Question in the Google Search Engine",
    checkpointer=MemorySaver()
)


while True:
    query = input("User : ")
    if query in ['exit','quit']:
        print("Good Bye")
        break

    resule = agent.invoke({'messages':[{'role':'user','content':query}]},{'configurable': {'thread_id': "123"}})
    print("AI : ",resule['messages'][-1].content)