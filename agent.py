from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage,AIMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.tools.retriever import create_retriever_tool 

loader = WebBaseLoader("https://python.langchain.com/v0.1/docs/expression_language/")
doc = loader.load()
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 200,
    chunk_overlap = 20          
)
splitDocs = splitter.split_documents(doc)

embedding = OpenAIEmbeddings()
vectorStore = FAISS.from_documents(splitDocs, embedding=embedding)
retriever = vectorStore.as_retriever(search_kwargs = {"k":3})

model = ChatOpenAI(
    model = "gpt-4o",
    temperature=0.7,
    verbose=True 
)

prompt = ChatPromptTemplate.from_messages([
    ("system","you are a friendly assistant called Max."),
    MessagesPlaceholder(variable_name="chat_History"),
    ("human","{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

search = TavilySearchResults()
retriever_tools = create_retriever_tool(
    retriever,
    "lcel_search",
    "use this tool when searching for information about Langchain Expression (LCEL) "
)
tools = [search,retriever_tools]         

agent = create_openai_functions_agent(
    llm=model,
    prompt=prompt,
    tools = tools
)

agentExecutor = AgentExecutor(
    agent = agent,
    tools = tools
)

def process_chat(agentExecutor,user_input,chat_History):
    response = agentExecutor.invoke({
        "input": user_input,
        "chat_History":chat_History
    })
    return response["output"]

if __name__ == "__main__":
    chat_History = []
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        response = process_chat(agentExecutor,user_input,chat_History)
        chat_History.append(HumanMessage(content=user_input))
        chat_History.append(AIMessage(content=response))
        print("Assistance:",response)