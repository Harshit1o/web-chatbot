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
        # Create a system prompt with detailed instructions
        system_prompt = f"""You are an AI assistant specialized in answering questions about website content.
You have been given content from the website {website_url}.
Your goal is to provide accurate, helpful, and concise answers based ONLY on the provided content.

GUIDELINES:
1. Answer ONLY based on the provided context. Do not use outside knowledge.
2. If the context contains the information needed to answer the question, provide a complete and accurate response.
3. If the context hints at an answer but is incomplete, provide what you can determine from the context and explain what's missing.
4. If the context doesn't contain relevant information for the question, respond with: "I don't have enough information from the website to answer that question."
5. Never make up information or pretend to know something that isn't in the provided context.
6. When appropriate, quote specific relevant parts from the context to support your answer.
7. Focus on providing factual information rather than opinions unless the question specifically asks for an opinion presented in the content.
8. Be detailed and thorough in your answers when the information is available in the context."""
        
        # Create a user prompt with the context and query
        user_prompt = f"""I need information from the following website content to answer a question:

CONTENT FROM {website_url}:
{context}

USER QUESTION:
{query}

Please provide an accurate answer based ONLY on the information in the content above."""
        
        # Use OpenAI GPT-4o-mini to generate a response
        # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
        # do not change this unless explicitly requested by the user
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Using GPT-4o-mini as requested
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,  # Lower temperature for more factual responses
            max_tokens=800,   # Allow for more detailed responses
        )
        
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating response: {str(e)}")
        return f"I encountered an error while generating a response: {str(e)}"
