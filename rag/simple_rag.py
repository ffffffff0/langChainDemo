import time
import sys
import os
import argparse

from evalute_rag import evaluate_rag
from init_tools import openai_model
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from helper_tools import (encode_pdf,
                          retrieve_context_per_question,
                          encode_csv,
                          show_context)

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

class SimpleRAG:
    def __init__(self, path, chunk_size=1000, chunk_overlap=200, n_retrieved=2):
        print("\n----------------- initialing simple rag retriever-----------------------")
        start_time = time.time()

        # self.vector_store = encode_pdf(path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        # load csv file 
        self.vector_store = encode_csv(path)
        self.time_records = {'Chunking': time.time() - start_time}

        print(f"Chunking Time: {self.time_records['Chunking']:.2f} seconds")
        self.chunks_query_retriever = self.vector_store.as_retriever(search_kwargs={"k": n_retrieved})
        self.llm = openai_model()

    def run(self, query):
        start_time = time.time()

        context = retrieve_context_per_question(query, self.chunks_query_retriever)
        self.time_records['Retrieval'] = time.time() - start_time
        print(f'Retrieval Time: {self.time_records['Retrieval']:.2f} seconds')

        show_context(context)

        # retrieval chain
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
            
        ])

        question_answer_chain = create_stuff_documents_chain(self.llm, prompt=prompt)
        rag_chain = create_retrieval_chain(
            retriever=self.chunks_query_retriever,
            combine_docs_chain=question_answer_chain
        )
        answer = rag_chain.invoke({"input": query})
        print(f'\n\nLLM Response: {answer["answer"]}\n\n')


def validate_args(args):
    if args.chunk_size <= 0:
        raise ValueError('chunk size must be a positive integer.')
    if args.chunk_overlap < 0:
        raise ValueError('chunk overlap must be a non-negative integer.')
    if args.n_retrieved <= 0:
        raise ValueError('number of retrieved chunks must be a positive integer.')
    
    return args

def parse_args():
    parser = argparse.ArgumentParser(description='Simple RAG Retriever')
    parser.add_argument('--chunk_size', type=int, default=1000, help='Size of each chunk')
    parser.add_argument('--chunk_overlap', type=int, default=200, help='Overlap between chunks')
    parser.add_argument('--n_retrieved', type=int, default=2, help='Number of chunks to retrieve')
    parser.add_argument('--query', type=str, default="which company does sheryl baxter work for?", 
                        required=True, help='Query to retrieve context for')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate the retriever with the provided query')
    
    args = parser.parse_args()
    return validate_args(args)


def main(args):
    rag_retriever = SimpleRAG(
        path='../example_data/customers-100.csv',
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        n_retrieved=args.n_retrieved
    )

    rag_retriever.run(args.query)

    if args.evaluate:
        evaluate_rag(rag_retriever.chunk_query_retriever)


if __name__ == "__main__":
    args = parse_args()
    main(args)
