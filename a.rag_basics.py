import os

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq

current_dir =os.path.dirname(os.path.abspath(__file__))
file_path =os.path.join(current_dir, "books", "file.txt")
persistent_directory = os.path.join(current_dir,"db", "cheoma_db")

#if chorma vector store already exists
if not os.path.exists(persistent_directory):
    print("Persistent directory does not exist. Initializing vector store...")
    
    #if chorma vector store alreadu exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"The file {file_path} does not exist. Please check the path."
    )

    #read text content from the file
    loader =TextLoader(file_path)
    documents=loader.load()

    #spilt the doc into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs= text_splitter.split_documents(documents)

    #display info about the split doc
    print("\n--- Document Chunks Information---")
    print(f"Number of document chunks: {len(docs)}")
    print(f"Sample chunk:\n{docs[0].page_content}\n")

    #create embeddings
    print("\n--- Creating embeddings ---")
    embeddings = GroqEmbeddings{
        model="text-embedding-3-small"
    }
    print("\n--- Finished creating embeddings ---")

    #creating vector store and persist it automatically
    print("\n--- Creating vector store ---")
    db = Chroma.from_documents(
        docs, embeddings, persist_directory=persistent_directory)
    print("\n--- Finished creating vector store ---")

else:
    print("Vector store already exists. No need to initialize.")
