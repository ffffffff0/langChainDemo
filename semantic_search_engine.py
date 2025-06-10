import os
from dotenv import load_dotenv

from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore


load_dotenv(
    dotenv_path=".env"
)

file_path = "./example_data/nke-10k-2023.pdf"
loader = PyPDFLoader(file_path)

docs = loader.load()

print("################ pdf docs info.")
print(len(docs))
print(f"{docs[0].page_content[:200]}\n")
print(docs[0].metadata)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

print("############### text splitter info.")
print(f"all splits: {len(all_splits)}")

# openai model
print("############## openai model.")
# Load the environment variables from the .env file
api_key = os.getenv("AI_ENDPOINT_API_KEY")
if api_key is None:
    raise ValueError("Can't get AI Endpoint token which generated from RDSec One Portal")
base_url = os.getenv("API_ENDPOINT")

embeddings = OpenAIEmbeddings(base_url= base_url, api_key=api_key, model="text-embedding-3-large")
vector_store = InMemoryVectorStore(embeddings)
ids = vector_store.add_documents(documents=all_splits)

results = vector_store.similarity_search(
    "How many distribution centers does Nike have in the US?"
)
print(f"How many distribution centers does Nike have in the US?\n, {results[0]}\n")

results = vector_store.similarity_search("When was Nike incorporated?")
print(f"When was Nike incorporated?\n, {results[0]}\n")

# Note that providers implement different scores; the score here
# is a distance metric that varies inversely with similarity.
results = vector_store.similarity_search_with_score("What was Nike's revenue in 2023?")
doc, score = results[0]
print(f"What was Nike's revenue in 2023?\n, score: {score},\n {doc}\n")

embedding = embeddings.embed_query("How were Nike's margins impacted in 2023?")
results = vector_store.similarity_search_by_vector(embedding)
print(f"How were Nike's margins impacted in 2023?\n {results[0]}\n")



