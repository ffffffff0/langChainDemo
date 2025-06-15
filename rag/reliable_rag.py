import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

from pydantic import BaseModel, Field
from typing import List
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from init_tools import openai_model, openai_embeddings
from langchain.output_parsers import PydanticOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser


embeddings = openai_embeddings()
urls = [
    "https://www.deeplearning.ai/the-batch/how-agents-can-improve-llm-performance/?ref=dl-staging-website.ghost.io",
    "https://www.deeplearning.ai/the-batch/agentic-design-patterns-part-2-reflection/?ref=dl-staging-website.ghost.io",
    "https://www.deeplearning.ai/the-batch/agentic-design-patterns-part-3-tool-use/?ref=dl-staging-website.ghost.io",
    "https://www.deeplearning.ai/the-batch/agentic-design-patterns-part-4-planning/?ref=dl-staging-website.ghost.io",
    "https://www.deeplearning.ai/the-batch/agentic-design-patterns-part-5-multi-agent-collaboration/?ref=dl-staging-website.ghost.io"
]

docs  = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=500,
    chunk_overlap=0
)

doc_splits = text_splitter.split_documents(docs_list)

vector_store = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag",
    embedding=embeddings
)

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

question = "what are the different kind of agentic design patterns?"
docs = retriever.invoke(question)
print(f'Title: {docs[0].metadata["title"]}\n\nSource: {docs[0].metadata["source"]}\n\nContent: {docs[0].page_content}\n')


# give a binary socre
class GradeDocuments(BaseModel):
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

llm = openai_model()
structured_llm_grader = llm.with_structured_output(GradeDocuments)
generate_system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", generate_system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)
retrieval_grader  = grade_prompt | structured_llm_grader

docs_to_use = []
for doc in docs:
    print(doc.page_content, '\n', '-'*100)
    res = retrieval_grader.invoke({"question": question, "document": doc.page_content})
    print(res, '\n')
    if res.binary_score == 'yes':
        docs_to_use.append(doc)

# generate result
generate_system = """You are an assistant for question-answering tasks. Answer the question based upon your knowledge. 
Use three-to-five sentences maximum and keep the answer concise."""
generate_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", generate_system),
        ("human", "Retrieved documents: \n\n <docs>{documents}</docs> \n\n User question: <question>{question}</question>"),
    ]
)

def format_docs(docs):
    return "\n".join(f"<doc{i+1}>:\nTitle:{doc.metadata['title']}\nSource:{doc.metadata['source']}\nContent:{doc.page_content}\n</doc{i+1}>\n" for i, doc in enumerate(docs))

rag_chain = generate_prompt | llm | StrOutputParser()
generation = rag_chain.invoke({"documents": format_docs(docs_to_use), "question": question})
print(f"generation: {generation}")

# check hallucinations
class GradeHallucinations(BaseModel):
    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no"
    )

structured_llm_grader = llm.with_structured_output(GradeHallucinations)
hallucination_system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
    Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""

hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", hallucination_system),
        ("human", "Set of facts: \n\n <facts>{documents}</facts> \n\n LLM generation: <generation>{generation}</generation>"),
    ]
)
hallucination_grader = hallucination_prompt | structured_llm_grader
response = hallucination_grader.invoke({
    "documents": format_docs(docs_to_use),
    "generation": generation
})

print(f"\n\nresponse: {response}\n\n")

# highlight used docs
class HighlightDocuments(BaseModel):
    id: List[str] = Field(
        ...,
        description="List of id of docs used to answers the question"
    )

    title: List[str] = Field(
        ...,
        description="List of titles used to answers the question"
    )

    source: List[str] = Field(
        ...,
        description="List of sources used to answers the question"
    )

    segment: List[str] = Field(
        ...,
        description="List of direct segments from used that answers the question"
    )


parser = PydanticOutputParser(pydantic_object=HighlightDocuments)
highlight_system ="""You are an advanced assistant for document search and retrieval. You are provided with the following:
1. A question.
2. A generated answer based on the question.
3. A set of documents that were referenced in generating the answer.

Your task is to identify and extract the exact inline segments from the provided documents that directly correspond to the content used to 
generate the given answer. The extracted segments must be verbatim snippets from the documents, ensuring a word-for-word match with the text 
in the provided documents.

Ensure that:
- (Important) Each segment is an exact match to a part of the document and is fully contained within the document text.
- The relevance of each segment to the generated answer is clear and directly supports the answer provided.
- (Important) If you didn't used the specific document don't mention it.

Used documents: <docs>{documents}</docs> \n\n User question: <question>{question}</question> \n\n Generated answer: <answer>{generation}</answer>

<format_instruction>
{format_instructions}
</format_instruction>
"""

highlight_prompt = PromptTemplate(
    template=highlight_system,
    input_variables=["documents", "question", "generation"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

doc_lookup = highlight_prompt | llm | parser
lookup_response = doc_lookup.invoke({
    "documents": format_docs(docs_to_use),
    "question": question,
    "generation": generation
})

for id, title, source, segment in zip(lookup_response.id, lookup_response.title, lookup_response.source, lookup_response.segment):
    print(f"ID: {id}\nTitle: {title}\nSource: {source}\nText Segment: {segment}\n{'-'*50}")

