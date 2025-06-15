import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

# load env var
load_dotenv()

api_key = os.getenv("AI_ENDPOINT_API_KEY")
if api_key is None:
    raise ValueError("Can't get AI Endpoint token which generated from RDSec One Portal")
base_url = os.getenv("API_ENDPOINT")

def openai_model(model='gpt-4.1') -> ChatOpenAI:
    # Initialize the ChatOpenAI object and the PromptTemplate in one line
    chat = ChatOpenAI(api_key=api_key, base_url=base_url, model=model)
    return chat


def openai_embeddings(model='text-embedding-3-large') -> OpenAIEmbeddings:
    embeddings = OpenAIEmbeddings(base_url= base_url, api_key=api_key, model=model)
    return embeddings


def qwen_model() -> ChatOpenAI:
    pass
