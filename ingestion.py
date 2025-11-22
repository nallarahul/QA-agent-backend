__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import shutil
from langchain_community.document_loaders import TextLoader, JSONLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Define paths
DB_PATH = "./vector_db"
UPLOAD_DIR = "./uploaded_docs"

class KnowledgeBase:
    def __init__(self):
        # Initialize Embedding Model (using standard MiniLM as per assignment suggestion)
        self.embedding_function = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

    def ingest_documents(self, source_files):
        """
        Processes uploaded files and stores them in ChromaDB.
        """
        documents = []
        
        # Ensure upload directory exists
        if not os.path.exists(UPLOAD_DIR):
            os.makedirs(UPLOAD_DIR)

        # 1. Save uploaded files temporarily to disk so loaders can read them
        saved_paths = []
        for file in source_files:
            file_path = os.path.join(UPLOAD_DIR, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            saved_paths.append(file_path)

        # 2. Load documents based on file type
        # We use simple text loading for MD/TXT/JSON as per Source 96
        for path in saved_paths:
            if path.endswith(".md") or path.endswith(".txt") or path.endswith(".json"):
                loader = TextLoader(path, encoding="utf-8")
                documents.extend(loader.load())
            # You can add PDF logic here if needed using PyMuPDF

        if not documents:
            return "No valid text documents found."

        # 3. Chunk the text (RecursiveCharacterTextSplitter) 
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)

        # 4. Store in ChromaDB 
        # Persist directory creates a folder on disk
        db = Chroma.from_documents(
            documents=chunks, 
            embedding=self.embedding_function,
            persist_directory=DB_PATH
        )
        db.persist() # Save to disk

        # Cleanup temp files
        for path in saved_paths:
            os.remove(path)
            
        return f"Successfully processed {len(chunks)} chunks from {len(source_files)} files."

    def get_retriever(self):
        """Returns the retriever object for the RAG pipeline."""
        db = Chroma(persist_directory=DB_PATH, embedding_function=self.embedding_function)
        return db.as_retriever(search_kwargs={"k": 4})