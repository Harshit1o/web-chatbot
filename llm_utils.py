import os
import numpy as np
from openai import OpenAI

def setup_openai():
    """
    Set up the OpenAI API client.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    
    return OpenAI(api_key=api_key)

# Initialize the OpenAI client
client = setup_openai()

def get_embeddings(text: str) -> list:
    """
    Get embeddings for text using OpenAI.
    
    Args:
        text (str): Text to get embeddings for
        
    Returns:
        list: Embedding vector for the text
    """
    try:
        # Use OpenAI to generate embeddings
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        
        # Extract and return the embedding values
        return response.data[0].embedding
    except Exception as e:
        print(f"Error generating embeddings: {str(e)}")
        raise

def generate_response(query: str, context: str, website_url: str) -> str:
    """
    Generate a response to a query using OpenAI GPT-4o-mini.
    
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
        
        # Use OpenAI GPT-4o-mini to generate a response
        # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
        # do not change this unless explicitly requested by the user
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Using GPT-4o-mini as requested
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions about website content."},
                {"role": "user", "content": prompt}
            ]
        )
        
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating response: {str(e)}")
        return f"I encountered an error while generating a response: {str(e)}"
