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

st.set_page_config(
    page_title="AskDocs AI",
    page_icon="🤖",
    layout="centered"  
)
st.markdown("""
<style>
    /* Hide sidebar completely */
    [data-testid="stSidebar"] { display: none; }
    
    /* Clean dark background */
    .stApp { background-color: #18181b; }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background-color: #27272a;
        border-radius: 12px;
        border: 1.5px dashed #3f3f46;
        padding: 8px;
    }
    
    /* Chat input */
    [data-testid="stChatInput"] textarea {
        background-color: #27272a !important;
        border-radius: 10px !important;
        color: #d4d4d8 !important;
    }

    /* All text */
    p, label, span, div { color: #d4d4d8; }

    /* Success box */
    .stAlert { 
        background-color: #27272a; 
        border: 0.5px solid #3f3f46;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="text-align: center; padding: 20px 0 16px;">
    <h1 style="font-size: 22px; font-weight: 500; color: #f4f4f5; margin: 0 0 8px;">
        AskDocs AI — Document Retrieval & Question Answering System
    </h1>
    <p style="font-size: 13px; color: #71717a; margin: 0;">
        Ask questions and get precise, context-aware answers grounded in your document.
    </p>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload your PDF",
    type="pdf",
    label_visibility="collapsed"
)

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
        search_kwargs={"k":6}
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


if uploaded_file:
    # save PDF temporarily
    temp_path= f"temp_{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    if "db" not in st.session_state:
        with st.spinner("Reading and indexing your PDF..."):
            st.session_state.db= load_and_index_pdf(temp_path)
        st.success("PDF loaded! Ask your questions below.")

        # store chat messages in session_state
    if "messages" not in st.session_state:
        st.session_state.messages= []

    for message in st.session_state.messages:  # display previous messages
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # chat input box at bottom
    if question:= st.chat_input("Ask a question about your document…"):
        # show user message immediately
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        # getting answer and show it
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = get_answer(
                    st.session_state.db, 
                    question,
                    "session1"
                )
            st.write(answer)
            st.session_state.messages.append({
                "role": "assistant", 
                "content": answer
            })

    if os.path.exists(temp_path):
        os.remove(temp_path)
else:
    st.markdown("""
    <div style="text-align: center; padding: 40px 0; color: #71717a;">
        <p style="font-size: 13px;">Start by uploading your PDF to explore its contents. </p>
    </div>
    """, unsafe_allow_html=True)