import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings

# setup llm
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

hugging_repo_id = "mistralai/Mistral-7B-Instruct-v0.3"

llm = HuggingFaceEndpoint(
    repo_id=hugging_repo_id,
    task="text-generation",
    temperature=0.5,
    max_length=512,
    huggingfacehub_api_token=HF_TOKEN
)

# custom prompt template
db_faiss_path= "vectorstore/db_faiss"
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer. 
Dont provide anything out of the given context

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

prompt = PromptTemplate(template= CUSTOM_PROMPT_TEMPLATE, input_variables=["context","question"])

#loading database 
embedding_model = HuggingFaceEmbeddings(model_name= "sentence-transformers/all-MiniLM-L6-v2")
db= FAISS.load_local(db_faiss_path, embedding_model, allow_dangerous_deserialization= True)

#Create QnA chain
qa_chain = RetrievalQA.from_chain_type(
    llm= llm ,
    chain_type="stuff",
    retriever= db.as_retriever(search_kwargs= {'k':5}),
    return_source_documents= True,
    chain_type_kwargs = {'prompt':prompt}
)


# invoke chain
user_query = input("Write your Query Here :")
response = qa_chain.invoke({'query': user_query})
print("Result:", response["result"])
print("Source of answer:", response["source_documents"])

