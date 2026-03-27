import os
import streamlit as st
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv

load_dotenv()
model = ChatGroq(model="llama-3.1-8b-instant")

store={}

CHROMA_DIR = "chroma_db"
def load_and_index_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks=splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    
    print("Creating embeddings..")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    print("Storing in ChromaDB..")
    db =Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=CHROMA_DIR,
        collection_metadata={"hnsw:space":"cosine"}
    )
    print("Done storing!")
    return db

def get_session_history(session_id):
    if session_id not in store:
        store[session_id]= InMemoryChatMessageHistory()
    return store[session_id]

def get_answer(db, question, session_id="session1"):
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k":3}
    )
    relevant_docs = retriever.invoke(question)
    context="\n\n".join([doc.page_content for doc in relevant_docs])
    model= ChatGroq(model="llama-3.3-70b-versatile")

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant that answers questions 
        based on the provided PDF content. Use the context below to answer.
        If the answer is not in the context, say so clearly.
         Context: {context}"""),
         MessagesPlaceholder(variable_name="history"),
         ("human", "{question}")
    ])
    chain =prompt | model | StrOutputParser()

    chain_with_history= RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="history"
    )
    response =chain_with_history.invoke(
        {"question": question,
        "context":context},
        config={"configurable": {"session_id":session_id}}
    )
    return response

if __name__ =="__main__":
    db = load_and_index_pdf("research_paper.pdf")
    print(get_answer(db, "What is the main finding of this research?"))
    print(get_answer(db, "can you elaborate on that?"))