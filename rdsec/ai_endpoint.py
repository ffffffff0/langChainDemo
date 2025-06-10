import os

from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv(
    dotenv_path="./.env"
)

# Load the environment variables from the .env file
api_key = os.getenv("AI_ENDPOINT_API_KEY")
if api_key is None:
    raise ValueError("Can't get AI Endpoint token which generated from RDSec One Portal")
base_url = os.getenv("API_ENDPOINT")
model = "gpt-4.1"

# Initialize the ChatOpenAI object and the PromptTemplate in one line
chat = ChatOpenAI(api_key=api_key, base_url=base_url, model=model)
prompt_template = PromptTemplate(template="Translate the following text to French,Chinese,Japanese in 3 lines: {text}",
                                 input_variables=["text"])

# Use the pipe operator to chain the prompt_template and chat, and invoke the chain in one line
result = (prompt_template | chat).invoke({"text": "Hello, how are you?"})

print(f"AI response: \n{result.content}\n")
print(f"Token usage:\n{result.response_metadata.get('token_usage')}")