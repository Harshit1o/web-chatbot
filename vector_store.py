import numpy as np
import faiss
from llm_utils import get_embeddings

def process_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> list:
    """
    Process and chunk the text content.
    
    Args:
        text (str): The text to process
        chunk_size (int): The size of each chunk in characters
        chunk_overlap (int): The overlap between chunks in characters
        
    Returns:
        list: A list of text chunks
    """
    if not text:
        return []
    
    # Clean the text
    text = text.replace('\n', ' ').replace('\r', ' ')
    
    # Split the text into chunks
    chunks = []
    for i in range(0, len(text), chunk_size - chunk_overlap):
        chunk = text[i:i + chunk_size]
        if len(chunk) > 50:  # Only keep chunks with meaningful content
            chunks.append(chunk)
    
    return chunks

def create_faiss_index(chunks: list) -> faiss.IndexFlatL2:
    """
    Create a FAISS index from text chunks.
    
    Args:
        chunks (list): List of text chunks
        
    Returns:
        faiss.IndexFlatL2: FAISS index
    """
    # Get embeddings for all chunks
    embeddings = []
    for chunk in chunks:
        embedding = get_embeddings(chunk)
        embeddings.append(embedding)
    
    # Convert to numpy array
    embeddings_array = np.array(embeddings).astype('float32')
    
    # Create FAISS index
    dimension = embeddings_array.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_array)
    
    return index

def search_similar_chunks(index, query_embedding, k=3):
    """
    Search for similar chunks in the FAISS index.
    
    Args:
        index: FAISS index
        query_embedding: Embedding of the query
        k (int): Number of similar chunks to retrieve
        
    Returns:
        list: Indices of the most similar chunks
    """
    # Convert query embedding to numpy array
    query_embedding_np = np.array([query_embedding]).astype('float32')
    
    # Search in the index
    distances, indices = index.search(query_embedding_np, k)
    
    return indices[0]
