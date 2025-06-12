from typing import List
import os

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.runnables import chain
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore

load_dotenv(
    dotenv_path=".env"
)
# load pdf file
file_path = "./example_data/nke-10k-2023.pdf"
loader = PyPDFLoader(file_path)

docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

# openai
api_key = os.getenv("AI_ENDPOINT_API_KEY")
if api_key is None:
    raise ValueError("Can't get AI Endpoint token which generated from RDSec One Portal")
base_url = os.getenv("API_ENDPOINT")
embeddings = OpenAIEmbeddings(base_url= base_url, api_key=api_key, model="text-embedding-3-large")
vector_store = InMemoryVectorStore(embeddings)
vector_store.add_documents(documents=all_splits)

@chain
def retriever(query: str) -> List[Document]:
    return vector_store.similarity_search(query, k=1)

result = retriever.batch(
    [
        "How many distribution centers does Nike have in the US?",
        "When was Nike incorporated?",
    ],
)
print(f"vector store search: {result}\n")

result = retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 1},
)
retriever.batch(
    [
        "How many distribution centers does Nike have in the US?",
        "When was Nike incorporated?",
    ],
)
print(f"similarity k=1 search: {result}\n")