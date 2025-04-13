import numpy as np
import faiss
import re
import logging
from llm_utils import get_embeddings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_and_normalize_text(text: str) -> str:
    """
    Clean and normalize text for better processing.
    
    Args:
        text (str): The raw text to clean
        
    Returns:
        str: Cleaned and normalized text
    """
    if not text:
        return ""
    
    # Replace multiple newlines with single newline
    text = re.sub(r'\n\s*\n', '\n', text)
    
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    
    # Normalize dashes and hyphens
    text = text.replace('–', '-').replace('—', '-')
    
    return text.strip()

def semantic_text_splitting(text: str) -> list:
    """
    Split text by semantic boundaries like paragraphs and section headers.
    
    Args:
        text (str): The text to split
        
    Returns:
        list: A list of semantically meaningful sections
    """
    if not text:
        return []
    
    # First try to split by double newlines (paragraphs)
    sections = re.split(r'\n\s*\n', text)
    
    # If we have very long paragraphs, further split them by single newlines
    refined_sections = []
    for section in sections:
        if len(section) > 1500:
            subsections = re.split(r'\n', section)
            refined_sections.extend(subsections)
        else:
            refined_sections.append(section)
    
    # Further split very long sections by sentences
    final_sections = []
    for section in refined_sections:
        if len(section) > 1500:
            # Simple sentence splitting - not perfect but works for most cases
            sentences = re.split(r'(?<=[.!?])\s+', section)
            
            # Group sentences together to form reasonable chunks
            current_chunk = ""
            for sentence in sentences:
                if len(current_chunk) + len(sentence) < 1500:
                    if current_chunk:
                        current_chunk += " " + sentence
                    else:
                        current_chunk = sentence
                else:
                    final_sections.append(current_chunk)
                    current_chunk = sentence
            
            if current_chunk:  # Add the last chunk
                final_sections.append(current_chunk)
        else:
            final_sections.append(section)
    
    # Filter out very short sections
    return [section for section in final_sections if len(section.strip()) > 50]

def process_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 250) -> list:
    """
    Process and chunk the text content using semantic boundaries when possible.
    
    Args:
        text (str): The text to process
        chunk_size (int): The maximum size of each chunk in characters
        chunk_overlap (int): The overlap between chunks in characters
        
    Returns:
        list: A list of text chunks
    """
    if not text:
        logger.warning("No text provided for processing")
        return []
    
    logger.info(f"Processing text of length: {len(text)} characters")
    
    # Clean and normalize the text
    cleaned_text = clean_and_normalize_text(text)
    
    # Try semantic splitting first
    semantic_chunks = semantic_text_splitting(cleaned_text)
    
    # If we have reasonable semantic chunks, use them
    if len(semantic_chunks) > 3:
        logger.info(f"Created {len(semantic_chunks)} semantic chunks")
        
        # Further split any extremely long chunks
        final_chunks = []
        for chunk in semantic_chunks:
            if len(chunk) > chunk_size + 500:  # If chunk is much larger than our target size
                # Split it using the simple sliding window approach
                for i in range(0, len(chunk), chunk_size - chunk_overlap):
                    sub_chunk = chunk[i:i + chunk_size]
                    if len(sub_chunk) > 50:
                        final_chunks.append(sub_chunk)
            else:
                final_chunks.append(chunk)
        
        logger.info(f"Final chunk count after processing: {len(final_chunks)}")
        return final_chunks
    
    # Fallback to simple chunking if semantic splitting didn't work well
    logger.info("Falling back to simple chunking")
    simple_chunks = []
    for i in range(0, len(cleaned_text), chunk_size - chunk_overlap):
        chunk = cleaned_text[i:i + chunk_size]
        if len(chunk) > 50:  # Only keep chunks with meaningful content
            simple_chunks.append(chunk)
    
    logger.info(f"Created {len(simple_chunks)} chunks using simple chunking")
    return simple_chunks

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

def search_similar_chunks(index, query_embedding, k=5, distance_threshold=1.0):
    """
    Search for similar chunks in the FAISS index using a distance threshold
    to ensure only relevant chunks are returned.
    
    Args:
        index: FAISS index
        query_embedding: Embedding of the query
        k (int): Maximum number of similar chunks to retrieve
        distance_threshold (float): Maximum distance to consider a chunk relevant
        
    Returns:
        tuple: (list of indices, list of distances) of the most similar chunks
    """
    # Convert query embedding to numpy array
    query_embedding_np = np.array([query_embedding]).astype('float32')
    
    # Search in the index for k*2 chunks to allow filtering
    k_search = min(k * 2, index.ntotal)  # Don't search for more chunks than exist
    if k_search == 0:
        logger.warning("No chunks available in the index")
        return [], []
        
    # Search in the index
    distances, indices = index.search(query_embedding_np, k_search)
    
    # Filter results by distance threshold and take the top k
    filtered_indices = []
    filtered_distances = []
    
    for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
        if len(filtered_indices) >= k:
            break
            
        # Lower distance means more relevant in L2 space
        if dist < distance_threshold:
            filtered_indices.append(idx)
            filtered_distances.append(dist)
    
    logger.info(f"Found {len(filtered_indices)} relevant chunks out of {k_search} searched")
    
    # If we have no chunks after filtering but have results, take the top ones anyway
    if not filtered_indices and len(indices[0]) > 0:
        logger.info("No chunks passed the distance threshold, using top results anyway")
        top_k = min(k, len(indices[0]))
        filtered_indices = indices[0][:top_k].tolist()
        filtered_distances = distances[0][:top_k].tolist()
    
    return filtered_indices, filtered_distances
