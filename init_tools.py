import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from openai import OpenAI as OpenAIClient

# load env var
load_dotenv()

api_key = os.getenv("AI_ENDPOINT_API_KEY")
if api_key is None:
    raise ValueError("Can't get AI Endpoint token which generated from RDSec One Portal")
base_url = os.getenv("API_ENDPOINT")

def llama_embeddings(model='text-embedding-3-large') -> OpenAIEmbedding:
    # Initialize the OpenAIEmbeddings object and the PromptTemplate in one line
    embeddings = OpenAIEmbedding(api_base=base_url, api_key=api_key, model=model)
    return embeddings

def llama_model(model='gpt-4.1') -> OpenAI:
    # Initialize the OpenAI object and the PromptTemplate in one line
    llm = OpenAI(api_key=api_key, api_base=base_url, model=model)
    return llm

def openai_model(model='gpt-4.1') -> ChatOpenAI:
    # Initialize the ChatOpenAI object and the PromptTemplate in one line
    chat = ChatOpenAI(api_key=api_key, base_url=base_url, model=model)
    return chat


def openai_embeddings(model='text-embedding-3-large') -> OpenAIEmbeddings:
    embeddings = OpenAIEmbeddings(base_url=base_url, api_key=api_key, model=model)
    return embeddings

def openai() -> OpenAIClient:
    client = OpenAIClient(api_key=api_key, base_url=base_url)
    return client


def qwen_model() -> ChatOpenAI:
    pass
