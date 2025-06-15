import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

from typing import Dict, Any, List
from init_tools import openai_model

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


def evaluate_rag(retriever, num_questions: int = 5) -> Dict[str, Any]:
    llm = openai_model()

    eval_prompt = PromptTemplate.from_template("""
        Evaluate the following retrieval results for the question.
        
        Question: {question}
        Retrieved Context: {context}
        
        Rate on a scale of 1-5 (5 being best) for:
        1. Relevance: How relevant is the retrieved information to the question?
        2. Completeness: Does the context contain all necessary information?
        3. Conciseness: Is the retrieved context focused and free of irrelevant information?
        
        Provide ratings in JSON format:
    """)

    eval_chain = (
        eval_prompt
        | llm
        | StrOutputParser()
    )

    question_gen_prompt = PromptTemplate.from_template(
        "Generate {num_questions} diverse test questions about climate chang:"
    )

    question_chain = question_gen_prompt | llm | StrOutputParser

    questions = question_chain.invoke({"num_questions": num_questions}).split('\n')

    results = []
    for question in questions:
        context = retriever.get_relevant_documents(question)
        context_text = '\n'.join([doc.page_content for doc in context])

        eval_result = eval_chain.invoke({
            'question': question,
            'context': context_text
        })

        results.append(eval_result)
    
    return {
        'questions': questions,
        'results': results,
        'average_socres': calculate_average_scores(results)
    }


def calculate_average_scores(results: List[Dict]) -> Dict[str, float]:
    pass


