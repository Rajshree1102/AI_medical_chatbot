import streamlit as st
import os
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
hugging_repo_id = "mistralai/Mistral-7B-Instruct-v0.3"


db_faiss_path = "vectorstore/db_faiss"
st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceBgeEmbeddings(model_name= "sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(db_faiss_path, embedding_model, allow_dangerous_deserialization=True)
    return db

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template= custom_prompt_template, input_variables=["context","question"])
    return prompt

def load_llm(huggingface_repo_id, HF_TOKEN):
    llm = HuggingFaceEndpoint(
    repo_id=hugging_repo_id,
    task="text-generation",
    temperature=0.5,
    max_length=512,
    huggingfacehub_api_token=HF_TOKEN
    )
    return llm
    

def main():
    st.title("Ask from Chatbot!")
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
        
    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])
    
    prompt = st.chat_input("Ask you query here")
    
    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role':'user', 'content':prompt})
        
        CUSTOM_PROMPT_TEMPLATE = """
                Use the pieces of information provided in the context to answer user's question.
                If you dont know the answer, just say that you dont know, dont try to make up an answer. 
                Dont provide anything out of the given context

                Context: {context}
                Question: {question}

                Start the answer directly. No small talk please.
                """
                
        
       
        
        try:
            vectorstore= get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store")
            
            qa_chain = RetrievalQA.from_chain_type(
                llm = load_llm(huggingface_repo_id=hugging_repo_id, HF_TOKEN= HF_TOKEN) ,
                chain_type="stuff",
                retriever= vectorstore.as_retriever(search_kwargs= {'k':5}),
                return_source_documents= True,
                chain_type_kwargs = {'prompt':set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )
            
            response = qa_chain.invoke({'query': prompt})
            
            result= response["result"]
            source_documents= response["source_documents"]
            result_to_show= result+"\nSource Docs:\n"+str(source_documents)
        
            #response= "Hi, I am here for help you!"
            st.chat_message('assistant').markdown(result_to_show)
            st.session_state.messages.append({'role':'assistant', 'content':result_to_show})
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
        
if __name__ == "__main__":
    main()
