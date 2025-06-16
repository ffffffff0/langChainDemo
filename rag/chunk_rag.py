import os 
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

import random
import time
import nest_asyncio
from init_tools import llama_model, llama_embeddings
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.core.prompts import PromptTemplate
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator, DatasetGenerator
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI

nest_asyncio.apply()

def evaluate_response_time_and_accuracy(chunk_size, eval_questions, eval_documents, 
                                        faithfulness_evaluator, relevancy_evaluator):
    
    total_response_time = 0
    total_faithfulness = 0
    total_relevancy = 0

    Settings.llm = llama_model(model='gpt-3.5-turbo')
    Settings.embed_model = llama_embeddings(model='text-embedding-3-large')

    # vector index
    splitter = SentenceSplitter(chunk_size=chunk_size)
    vector_index = VectorStoreIndex.from_documents(eval_documents, transformations=[splitter])

    # build query engine
    query_engine = vector_index.as_query_engine(
        similarity_top_k=5,
    )
    num_questions = len(eval_questions)

    for question in eval_questions:
        start_time = time.time()
        response_vector = query_engine.query(question)
        elapsed_time = time.time() - start_time

        faithfulness_result = faithfulness_evaluator.evaluate_response(
            response=response_vector
        ).passing
        relevancy_result = relevancy_evaluator.evaluate_response(
            query=question,
            response=response_vector
        ).passing

        total_response_time += elapsed_time
        total_faithfulness += faithfulness_result
        total_relevancy += relevancy_result
    
    avg_response_time = total_response_time / num_questions
    avg_faithfulness = total_faithfulness / num_questions
    avg_relevancy = total_relevancy / num_questions

    return avg_response_time, avg_faithfulness, avg_relevancy


class RAGEvaluator:
    def __init__(self, data_dir, num_eval_questions, chunk_sizes):
        self.data_dir = data_dir
        self.num_eval_questions = num_eval_questions
        self.chunk_sizes = chunk_sizes
        self.documents = self.load_documents()

        self.eval_questions = self.generate_eval_questions()

        self.llm_gpt4 = llama_model(model='gpt-4o')

        self.faithfulness_evaluator = self.create_faithfulness_evaluator()
        self.relevancy_evaluator = self.create_relevancy_evaluator()
    
    def load_documents(self):
        return SimpleDirectoryReader(self.data_dir).load_data()
    
    def generate_eval_questions(self):
        eval_documents = self.documents[0:20]
        data_generator = DatasetGenerator.from_documents(documents=eval_documents, llm=llama_model(model='gpt-4o'))

        eval_questions = data_generator.generate_questions_from_nodes()
        return random.sample(eval_questions, self.num_eval_questions)
    
    def create_relevancy_evaluator(self):
        return RelevancyEvaluator(
            llm=self.llm_gpt4,
        )

    
    def create_faithfulness_evaluator(self):
        faithfulness_evaluator = FaithfulnessEvaluator(
            llm=self.llm_gpt4,
        )

        faithfulness_new_prompt_template = PromptTemplate(""" Please tell if a given piece of information is directly supported by the context.
            You need to answer with either YES or NO.
            Answer YES if any part of the context explicitly supports the information, even if most of the context is unrelated. If the context does not explicitly support the information, answer NO. Some examples are provided below.

            Information: Apple pie is generally double-crusted.
            Context: An apple pie is a fruit pie in which the principal filling ingredient is apples.
            Apple pie is often served with whipped cream, ice cream ('apple pie à la mode'), custard, or cheddar cheese.
            It is generally double-crusted, with pastry both above and below the filling; the upper crust may be solid or latticed (woven of crosswise strips).
            Answer: YES

            Information: Apple pies taste bad.
            Context: An apple pie is a fruit pie in which the principal filling ingredient is apples.
            Apple pie is often served with whipped cream, ice cream ('apple pie à la mode'), custard, or cheddar cheese.
            It is generally double-crusted, with pastry both above and below the filling; the upper crust may be solid or latticed (woven of crosswise strips).
            Answer: NO

            Information: Paris is the capital of France.
            Context: This document describes a day trip in Paris. You will visit famous landmarks like the Eiffel Tower, the Louvre Museum, and Notre-Dame Cathedral.
            Answer: NO

            Information: {query_str}
            Context: {context_str}
            Answer:
        """)
        faithfulness_evaluator.update_prompts({"your_prompt_key": faithfulness_new_prompt_template})
        return faithfulness_evaluator
    
    def run(self):
        for chunk_size in self.chunk_sizes:
            avg_response_time, avg_faithfulness, avg_relevancy = evaluate_response_time_and_accuracy(
                chunk_size,
                self.eval_questions,
                self.documents[0:20],
                self.faithfulness_evaluator,
                self.relevancy_evaluator

            )

            print(f"Chunk Size: {chunk_size} - Average Response Time: {avg_response_time:.2f}s, "
                  f"Average Faithfulness: {avg_faithfulness:.2f}, Average Relevancy: {avg_relevancy:.2f}")


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description='RAG Evaluator')
    parser.add_argument('--data_dir', type=str, default='../example_data', help='Directory containing documents for evaluation')
    parser.add_argument('--num_eval_questions', type=int, default=5, help='Number of evaluation questions to generate')
    parser.add_argument('--chunk_sizes', type=int, nargs='+', default=[128, 256, 512], 
                        help='Chunk sizes to evaluate')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    evaluator = RAGEvaluator(
        data_dir=args.data_dir,
        num_eval_questions=args.num_eval_questions,
        chunk_sizes=args.chunk_sizes
    )
    
    evaluator.run()