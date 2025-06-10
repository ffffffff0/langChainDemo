from typing import List
import os

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.runnables import chain
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings

load_dotenv(
    dotenv_path=".env"
)
api_key = os.getenv("AI_ENDPOINT_API_KEY")
if api_key is None:
    raise ValueError("Can't get AI Endpoint token which generated from RDSec One Portal")
base_url = os.getenv("API_ENDPOINT")
embeddings = OpenAIEmbeddings(base_url= base_url, api_key=api_key, model="text-embedding-3-large")
vector_store = InMemoryVectorStore(embeddings)

@chain
def retriever(query: str) -> List[Document]:
    return vector_store.similarity_search(query, k=1)

retriever.batch(
    [
        "How many distribution centers does Nike have in the US?",
        "When was Nike incorporated?",
    ],
)

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 1},
)
retriever.batch(
    [
        "How many distribution centers does Nike have in the US?",
        "When was Nike incorporated?",
    ],
)