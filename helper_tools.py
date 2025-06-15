import os
import sys
import pandas as pd
import faiss

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

from init_tools import openai_embeddings
from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore

def replace_t_with_space(list_of_docs):
    # replace tabs with spaces
    for doc in list_of_docs:
        doc.page_content = doc.page_content.replace("\t", " ")
    
    return list_of_docs

def encode_pdf(path, chunk_size=1000, chunk_overlap=200):
    """
    Load a PDF file, split it into chunks, and create a vector store.
    Args:
        path (str): Path to the PDF file.
        chunk_size (int): Size of each chunk.
        chunk_overlap (int): Overlap between chunks.
    Returns:
        vector_store (FAISS): A vector store containing the chunks of the PDF.
    """

    loader = PyPDFLoader(path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )

    texts = text_splitter.split_documents(docs)
    cleaned_texts = replace_t_with_space(texts)

    embeddings = openai_embeddings()
    vector_store = FAISS.from_documents(cleaned_texts, embeddings)

    return vector_store


def retrieve_context_per_question(query, chunks_query_retriever):
    docs = chunks_query_retriever.get_relevant_documents(query)
    context = [doc.page_content for doc in docs]

    return context


def show_context(context):
    for i, c in enumerate(context):
        print(f'Context {i + i}')
        print(c)
        print('\n')


def encode_csv(path):
    data = pd.read_csv(path)
    print(f'csv head: {data.head()}')

    loader = CSVLoader(file_path=path)
    docs = loader.load_and_split()

    embeddings = openai_embeddings()
    idx = faiss.IndexFlatL2(len(embeddings.embed_query(" ")))
    vector_store = FAISS(
        embedding_function=embeddings,
        index=idx,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={}
    )

    vector_store.add_documents(docs)
    return vector_store


















