import os
import openai
import sys
import langchain
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv, find_dotenv
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

# Load the .env file
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key  = os.environ['OPENAI_API_KEY']
# langchain.debug=True


# Loading the existing vectore database
embedding = OpenAIEmbeddings()
erasmus_vectordb = Chroma(persist_directory='./db/erasmus_page', embedding_function=embedding)


# Setting a QA Chaing
# Constant
llm_name = "gpt-3.5-turbo-1106"
n_documents = 3
chain_type= "stuff"

# Retriver
retriever = erasmus_vectordb.as_retriever(search_type="similarity", search_kwargs={"k": n_documents})

# llm
llm=ChatOpenAI(model_name=llm_name, temperature=0)

# Memory 
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)


#ConversationalRetrievalChain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm,
    chain_type=chain_type,
    retriever=retriever,
    memory=memory,
    return_source_documents=False,
    return_generated_question=False,
)

# Start the conversation loop
done = False
while not done:
    Query = input("-->: ")
    if Query == "exit":
        done = True
    else:
        result = qa_chain.invoke({'question': Query})
        answer = result['answer']
        print(answer)