import os
import google.generativeai as genai
import numpy as np

def setup_gemini():
    """
    Set up the Gemini API client.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables")
    
    genai.configure(api_key=api_key)

# Initialize the Gemini API
setup_gemini()

def get_embeddings(text: str) -> list:
    """
    Get embeddings for text using Gemini.
    
    Args:
        text (str): Text to get embeddings for
        
    Returns:
        list: Embedding vector for the text
    """
    try:
        # Use Gemini to generate embeddings
        embedding_model = genai.get_generative_model("embedding-001")
        embedding = embedding_model.embed_content(text)
        
        # Extract and return the embedding values
        return embedding.embedding
    except Exception as e:
        print(f"Error generating embeddings: {str(e)}")
        raise

def generate_response(query: str, context: str, website_url: str) -> str:
    """
    Generate a response to a query using Gemini.
    
    Args:
        query (str): User's question
        context (str): Relevant context from the website
        website_url (str): URL of the website
        
    Returns:
        str: Generated response
    """
    try:
        # Create a prompt with the context and query
        prompt = f"""
        I'm going to provide you with content from the website {website_url} and a question about this content.
        
        CONTENT:
        {context}
        
        QUESTION:
        {query}
        
        Please answer the question based ONLY on the provided content. If the information needed to answer the 
        question is not in the provided content, say "I don't have enough information from the website to answer 
        that question" instead of making up an answer. Your answer should be helpful, concise, and accurate.
        """
        
        # Use Gemini to generate a response
        generation_model = genai.get_generative_model("gemini-pro")
        response = generation_model.generate_content(prompt)
        
        return response.text
    except Exception as e:
        print(f"Error generating response: {str(e)}")
        return f"I encountered an error while generating a response: {str(e)}"
