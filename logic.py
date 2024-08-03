from langchain_openai import OpenAI, OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.retrievers import ContextualCompressionRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain.retrievers.document_compressors import (
    LLMChainExtractor,
    EmbeddingsFilter,
    LLMChainFilter,
    DocumentCompressorPipeline,
)
import json

# -------------
import os
import openai
from dotenv import load_dotenv, find_dotenv


# Load and set OpenAI API Key
try:
    _ = load_dotenv(find_dotenv()) # read local .env file
except:
    pass
openai.api_key = os.environ['OPENAI_API_KEY']
# -------------


embedding = OpenAIEmbeddings()

ytu_vectordb = Chroma(persist_directory='./db/ytu_website', embedding_function=embedding)
end_vectordb = Chroma(persist_directory='./db/end_website', embedding_function=embedding)
erasmus_vectordb = Chroma(persist_directory='./db/erasmus_website', embedding_function=embedding)



# Filters and Compressers
llm = OpenAI(temperature=0)

relevant_filter = LLMChainFilter.from_llm(llm)
embeddings_filter = EmbeddingsFilter(embeddings=OpenAIEmbeddings(), similarity_threshold=0.85)
compressor = LLMChainExtractor.from_llm(llm)
pipeline_compressor = DocumentCompressorPipeline(
    transformers=[embeddings_filter, relevant_filter]
)

base_compressor = pipeline_compressor


# Build prompt
erasmus_template = PromptTemplate.from_template(
"""You are Yildiz Technical University' helpful chatbot, called 'YTU Chatbot'.\
You have a funny and energetic spirit. The students, staff from this university \
will ask you specifically about Erasmus Program. Respond to them joyfully and helpfully.
For every question asked, you will be provided with a context to assist you in answering.\
The context has been retrieved from the scraped University Erasmus Page.
Questions and context might be in Turkish.
Append the url of the source you have used to the end of the answer.

Use the following pieces of context to answer the question at the end. \
If you don't know the answer, just say that you don't know,\
don't try to make up an answer. Give a detailed answer.
{context}
Question: {query}
Helpful Answer:""")

end_template = PromptTemplate.from_template(
"""You are Yildiz Technical University' helpful chatbot, called 'YTU Chatbot'.\
You have a funny and energetic spirit. The students, staff from this university \
will ask you specifically about Industrial Engineering major. Respond to them joyfully\
and helpfully. For every question asked, you will be provided with a context to\
assist you in answering. The context has been retrieved from the\
scraped University Industrial Engineering Page.
Questions and context might be in Turkish.
Append the url of the source you have used to the end of the answer.

Use the following pieces of context to answer the question at the end. \
If you don't know the answer, just say that you don't know,\
don't try to make up an answer. 
{context}
Question: {query}
Helpful Answer:""")

ytu_template = PromptTemplate.from_template(
"""You are Yildiz Technical University' helpful chatbot, called 'YTU Chatbot'.\
You have a funny and energetic spirit. The students, staff from this university \
will ask you various questions. Respond to them joyfully and funnyly.
For every question asked, you will be provided with a context to assist you in answering.\
The context has been retrieved from the scraped university's general page.
Questions and contexts might be in Turkish.
Append the url of the source you have used to the end of the answer.
REMEMMBER! Always greet them at the beggining of the Helpful Answer.

Use the following pieces of context to answer the question at the end. \
If you don't know the answer, just say that you don't know,\
don't try to make up an answer. Give a detailed answer. 
{context}
Question: {query}
Helpful Answer:""")



end_compression_retriever = ContextualCompressionRetriever(
    base_compressor=base_compressor,
    base_retriever=end_vectordb.as_retriever(k=1)
)
ytu_compression_retriever = ContextualCompressionRetriever(
    base_compressor=base_compressor,
    base_retriever=ytu_vectordb.as_retriever()
)
erasmus_compression_retriever = ContextualCompressionRetriever(
    base_compressor=base_compressor,
    base_retriever=erasmus_vectordb.as_retriever()
)

# END chain
end_qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=end_compression_retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": end_template}
)
# Erasmus chain
erasmus_qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=erasmus_vectordb.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt": erasmus_template}
)

# YTU chain
ytu_qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=ytu_compression_retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": ytu_template}
)


gpt_llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
chain = (
    PromptTemplate.from_template(
        """You are yildiz technical university chatbot. \
        You will be given a student question below, \
        classify it as either being about `General`, `Major`, or `Erasmus`.
        The `Major` classification will be if the student asks questions \
        related to their major. The `Erasmus` classification if they only \
        asked about Erasmus. The `General` classification will be if they asked\
        generally about university and stuff.

Do not respond with more than one word.

<question>
{query}
</question>

Classification:"""
    )
    | gpt_llm
    | StrOutputParser()
)

#-----------------------------------------------------



#-----------------------------------------------------

# def route(info):
#     if "erasmus" in info["topic"].lower():
#         return erasmus_qa_chain
#     elif "major" in info["topic"].lower():
#         return end_qa_chain
#     else:
#         return ytu_qa_chain

# #-----------------------------------------------------


# ytu_chatbot_chain = {"topic": chain, "query": lambda x: x["query"]} | RunnableLambda(route)


#------------------------- DEBUG CODE ----------------------------

def route(info):
    print(f"DEBUG: route function received: {json.dumps(info, indent=2)}")
    if "erasmus" in info.get("topic", "").lower():
        print("DEBUG: Routing to erasmus_qa_chain")
        return erasmus_qa_chain
    elif "major" in info.get("topic", "").lower():
        print("DEBUG: Routing to end_qa_chain")
        return end_qa_chain
    else:
        print("DEBUG: Routing to ytu_qa_chain")
        return ytu_qa_chain

# Update the chain to use 'query' consistently and add debug prints
ytu_chatbot_chain = (
    {
        "topic": lambda x: print(f"DEBUG: Topic input: {x}") or chain.invoke(x),
        "query": lambda x: print(f"DEBUG: Query input: {x}") or x["query"]
    }
    | RunnableLambda(route)
)

# Wrap the chain with a debug function
def debug_chain(input_dict):
    print(f"DEBUG: Chain input: {json.dumps(input_dict, indent=2)}")
    result = ytu_chatbot_chain.invoke(input_dict)
    print(f"DEBUG: Chain output: {json.dumps(result, indent=2)}")
    return result

# Replace ytu_chatbot_chain with the debug version
ytu_chatbot_chain = debug_chain
