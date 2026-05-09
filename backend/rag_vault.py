import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader, DataFrameLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import pandas as pd

class RAGVault:
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        # Use a small, fast local embedding model
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_store = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
            collection_name="automotive_requirements"
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", " "]
        )

    async def ingest_pdf(self, file_path: str):
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        splits = self.text_splitter.split_documents(documents)
        self.vector_store.add_documents(splits)
        return len(splits)

    async def ingest_excel(self, file_path: str):
        df = pd.read_excel(file_path)
        # Assuming requirements are in a column named 'Description' or 'Text'
        text_col = next((c for c in df.columns if c.lower() in ['description', 'text', 'requirement']), df.columns[0])
        loader = DataFrameLoader(df, page_content_column=text_col)
        documents = loader.load()
        splits = self.text_splitter.split_documents(documents)
        self.vector_store.add_documents(splits)
        return len(splits)

    def search_context(self, query: str, k: int = 3) -> str:
        results = self.vector_store.similarity_search(query, k=k)
        context = "\n---\n".join([doc.page_content for doc in results])
        return context

    def get_stats(self):
        # Basic stats about the collection
        count = self.vector_store._collection.count()
        return {"total_chunks": count}
