import os 
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

from init_tools import openai
import pymupdf
import numpy as np
from tqdm import tqdm
import json
class RAG:
    def __init__(self, pdf_path, question_path):
        self.pdf_path = pdf_path
        self.client = openai()

        # Load the validation data from a JSON file
        with open(question_path) as f:
            self.data = json.load(f)

        # generate response
        # Define the system prompt for the AI assistant
        self.system_prompt = """You are an AI assistant that strictly answers based on the given context. 
        If the answer cannot be derived directly from the provided context, 
        respond with: 'I do not have enough information to answer that.'"""

        self.FAITHFULNESS_PROMPT_TEMPLATE = """
        Evaluate the faithfulness of the AI response compared to the true answer.
        User Query: {question}
        AI Response: {response}
        True Answer: {true_answer}

        Faithfulness measures how well the AI response aligns with facts in the true answer, without hallucinations.

        INSTRUCTIONS:
        - Score STRICTLY using only these values:
            * {full} = Completely faithful, no contradictions with true answer
            * {partial} = Partially faithful, minor contradictions
            * {none} = Not faithful, major contradictions or hallucinations
        - Return ONLY the numerical score ({full}, {partial}, or {none}) with no explanation or additional text.
        """
        self.RELEVANCY_PROMPT_TEMPLATE = """
        Evaluate the relevancy of the AI response to the user query.
        User Query: {question}
        AI Response: {response}

        Relevancy measures how well the response addresses the user's question.

        INSTRUCTIONS:
        - Score STRICTLY using only these values:
            * {full} = Completely relevant, directly addresses the query
            * {partial} = Partially relevant, addresses some aspects
            * {none} = Not relevant, fails to address the query
        - Return ONLY the numerical score ({full}, {partial}, or {none}) with no explanation or additional text.
        """

    # extract text from pdf
    def extract_text_from_pdf(self):
        mypdf = pymupdf.open(self.pdf_path)
        all_text = ""

        for page in mypdf:
            # Extract text from the current page and add spacing
            all_text += page.get_text("text") + " "
        return all_text.strip()
    
    # Chunking the Extracted Text
    def chunk_text(self, text, n, overlap):
        chunks = []

        for i in range(0, len(text), n - overlap):
            # Append a chunk of text from the current index to the index + chunk size
            # ensure every chunk have context
            chunks.append(text[i:i + n])
        return chunks

    def generate_chunk_header(self, chunk, model="gpt-4o"):
        # Define the system prompt to guide the AI's behavior
        system_prompt = "Generate a concise and informative title for the given text."
        
        # Generate a response from the AI model based on the system prompt and text chunk
        response = self.client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": chunk}
            ]
        )

        # Return the generated header/title, stripping any leading/trailing whitespace
        return response.choices[0].message.content.strip()
    

    def chunk_text_with_headers(self, text, n, overlap):
        chunks = []  # Initialize an empty list to store chunks

        # Iterate through the text with the specified chunk size and overlap
        for i in range(0, len(text), n - overlap):
            chunk = text[i:i + n]  # Extract a chunk of text
            header = self.generate_chunk_header(chunk)  # Generate a header for the chunk using LLM
            chunks.append({"header": header, "text": chunk})  # Append the header and chunk to the list

        return chunks  # Return the list of chunks with headers
    
    # create embedding
    def create_embeddings(self, texts, model="text-embedding-3-large"):
        response = self.client.embeddings.create(input=texts, model=model)
        return [np.array(embedding.embedding) for embedding in response.data]

    def create_embeddings_v2(self, text, model="text-embedding-3-large"):
        response = self.client.embeddings.create(
            model=model,
            input=text
        )
        return response  # Return the response containing the embeddings
    

    # Performing Semantic Search
    def cosine_similarity(self, v1, v2):
        # Compute the dot product of the two vectors
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    def context_enriched_search(self, query, text_chunks, embeddings, k=1, context_size=1):
        query_embedding = self.create_embeddings_v2(query).data[0].embedding
        similarity_scores = []
        # Compute similarity scores between query and each text chunk embedding
        for i, chunk_embedding in enumerate(embeddings):
            # Calculate cosine similarity between the query embedding and current chunk embedding
            similarity_score = self.cosine_similarity(np.array(query_embedding), np.array(chunk_embedding.embedding))
            # Store the index and similarity score as a tuple
            similarity_scores.append((i, similarity_score))
        
        # Sort chunks by similarity score in descending order (highest similarity first)
        similarity_scores.sort(key=lambda x: x[1], reverse=True)
        # Get the index of the most relevant chunk
        top_index = similarity_scores[0][0]
        # Define the range for context inclusion
        # Ensure we don't go below 0 or beyond the length of text_chunks
        start = max(0, top_index - context_size)
        end = min(len(text_chunks), top_index + context_size + 1)
        # Return the relevant chunk along with its neighboring context chunks
        return [text_chunks[i] for i in range(start, end)]

    def retrieve_relevant_chunks(self, query, text_chunks, chunk_embeddings, k=5):
        query_embedding = self.create_embeddings([query])[0]
        similarities = [self.cosine_similarity(query_embedding, emb) for emb in chunk_embeddings]
        # Get the indices of the top-k most similar chunks
        top_indices = np.argsort(similarities)[-k:][::-1]
        # Return the top-k most relevant text chunks
        return [text_chunks[i] for i in top_indices]
    

    def generate_response(self, query, system_prompt, retrieved_chunks, model="gpt-4o"):
        # Combine retrieved chunks into a single context string
        context = "\n".join([f"Context {i+1}:\n{chunk}" for i, chunk in enumerate(retrieved_chunks)])
        # Create the user prompt by combining the context and the query
        user_prompt = f"{context}\n\nQuestion: {query}"
        # Generate the AI response using the specified model
        response = self.client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )

        # Return the content of the AI response
        return response.choices[0].message.content
    

    # Define strict evaluation prompt templates
    def evaluate_response(self, question, response, true_answer):
            # Format the evaluation prompts
            faithfulness_prompt = self.FAITHFULNESS_PROMPT_TEMPLATE.format(
                    question=question, 
                    response=response, 
                    true_answer=true_answer,
                    full=SCORE_FULL,
                    partial=SCORE_PARTIAL,
                    none=SCORE_NONE
            )
            
            relevancy_prompt = self.RELEVANCY_PROMPT_TEMPLATE.format(
                    question=question, 
                    response=response,
                    full=SCORE_FULL,
                    partial=SCORE_PARTIAL,
                    none=SCORE_NONE
            )

            # Request faithfulness evaluation from the model
            faithfulness_response = self.client.chat.completions.create(
                model="gpt-4o",
                    temperature=0,
                    messages=[
                            {"role": "system", "content": "You are an objective evaluator. Return ONLY the numerical score."},
                            {"role": "user", "content": faithfulness_prompt}
                    ]
            )
            
            # Request relevancy evaluation from the model
            relevancy_response = self.client.chat.completions.create(
                    model="gpt-4o",
                    temperature=0,
                    messages=[
                            {"role": "system", "content": "You are an objective evaluator. Return ONLY the numerical score."},
                            {"role": "user", "content": relevancy_prompt}
                    ]
            )
            
            # Extract scores and handle potential parsing errors
            try:
                    faithfulness_score = float(faithfulness_response.choices[0].message.content.strip())
            except ValueError:
                    print("Warning: Could not parse faithfulness score, defaulting to 0")
                    faithfulness_score = 0.0
                    
            try:
                    relevancy_score = float(relevancy_response.choices[0].message.content.strip())
            except ValueError:
                    print("Warning: Could not parse relevancy score, defaulting to 0")
                    relevancy_score = 0.0

            return faithfulness_score, relevancy_score
    
    def chunk_run(self):
        # Extract text from the PDF file
        extracted_text = self.extract_text_from_pdf(pdf_path)
        # Print the first 500 characters of the extracted text
        print(extracted_text[:500])

        # Define different chunk sizes to evaluate
        chunk_sizes = [128, 256, 512]
        # Create a dictionary to store text chunks for each chunk size
        text_chunks_dict = {size: self.chunk_text(extracted_text, size, size // 5) for size in chunk_sizes}
        # Print the number of chunks created for each chunk size
        for size, chunks in text_chunks_dict.items():
            print(f"Chunk Size: {size}, Number of Chunks: {len(chunks)}")
        

        # Generate embeddings for each chunk size
        # Iterate over each chunk size and its corresponding chunks in the text_chunks_dict
        chunk_embeddings_dict = {size: self.create_embeddings(chunks) for size, 
                                 chunks in tqdm(text_chunks_dict.items(), desc="Generating Embeddings")}
        

        query = self.data[0]['question']
        retrieved_chunks_dict = {size: self.retrieve_relevant_chunks(query, text_chunks_dict[size], 
                                                                     chunk_embeddings_dict[size]) for size in chunk_sizes}
        # Print retrieved chunks for chunk size 256
        print(retrieved_chunks_dict[256])

        # Evaluating the AI Response
        # Define evaluation scoring system constants
        SCORE_FULL = 1.0     # Complete match or fully satisfactory
        SCORE_PARTIAL = 0.5  # Partial match or somewhat satisfactory
        SCORE_NONE = 0.0     # No match or unsatisfactory
        # True answer for the first validation data
        true_answer = self.data[0]['answer']
        # Evaluate response for chunk size 256 and 128
        faithfulness, relevancy = self.evaluate_response(query, ai_responses_dict[256], true_answer)
        faithfulness2, relevancy2 = self.evaluate_response(query, ai_responses_dict[128], true_answer)
        # print the evaluation scores
        print(f"Faithfulness Score (Chunk Size 256): {faithfulness}")
        print(f"Relevancy Score (Chunk Size 256): {relevancy}")

        print(f"\n")

        print(f"Faithfulness Score (Chunk Size 128): {faithfulness2}")
        print(f"Relevancy Score (Chunk Size 128): {relevancy2}")



# Define the path to the PDF file
pdf_path = "../example_data/Understanding_Climate_Change.pdf"

# Chunk the extracted text into segments of 1000 characters with an overlap of 200 characters
text_chunks = chunk_text(extracted_text, 1000, 200)
# Print the number of text chunks created
print("Number of text chunks:", len(text_chunks))
# Print the first text chunk
print("\nFirst text chunk:")
print(text_chunks[0])


# Create embeddings for the text chunks
response = create_embeddings_v2(text_chunks)


# Extract the first query from the validation data
query = data[3]['question']
# Retrieve relevant chunks for each chunk size
retrieved_chunks_dict = {size: retrieve_relevant_chunks(query, text_chunks_dict[size], chunk_embeddings_dict[size]) for size in chunk_sizes}
# Print retrieved chunks for chunk size 256
print(retrieved_chunks_dict[256])
# Extract the first question from the dataset to use as our query
query = data[0]['question']

# Retrieve the most relevant chunk and its neighboring chunks for context
# Parameters:
# - query: The question we're searching for
# - text_chunks: Our text chunks extracted from the PDF
# - response.data: The embeddings of our text chunks
# - k=1: Return the top match
# - context_size=1: Include 1 chunk before and after the top match for context
top_chunks = context_enriched_search(query, text_chunks, response.data, k=1, context_size=1)
# Print the query for reference
print("Query:", query)
# Print each retrieved chunk with a heading and separator
for i, chunk in enumerate(top_chunks):
    print(f"Context {i + 1}:\n{chunk}\n=====================================")


# Generate AI responses for each chunk size
ai_responses_dict = {size: generate_response(query, system_prompt, retrieved_chunks_dict[size]) for size in chunk_sizes}
# Print the response for chunk size 256
print(ai_responses_dict[256])


if __name__ == "__main__":
    
    evaluator = RAG()
    
    evaluator.chunk_run()
