import streamlit as st
import os
from scraper import scrape_website
from vector_store import process_text, create_faiss_index, search_similar_chunks
from llm_utils import get_embeddings, generate_response
import database as db
import time

# Set page title and configuration
st.set_page_config(
    page_title="Website Chatbot Generator",
    page_icon="💬",
    layout="wide"
)

# Application title and description
st.title("AI Chatbot Generator for Websites")
st.markdown("Input a website URL to create a custom chatbot that can answer questions about the content.")

# Initialize session state variables
if "chatbot_ready" not in st.session_state:
    st.session_state.chatbot_ready = False
if "index" not in st.session_state:
    st.session_state.index = None
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "website_url" not in st.session_state:
    st.session_state.website_url = ""
if "website_id" not in st.session_state:
    st.session_state.website_id = None
if "chat_session_id" not in st.session_state:
    st.session_state.chat_session_id = None
if "debug_mode" not in st.session_state:
    st.session_state.debug_mode = False
if "content_stats" not in st.session_state:
    st.session_state.content_stats = {}
if "last_query_info" not in st.session_state:
    st.session_state.last_query_info = {}
if "last_context" not in st.session_state:
    st.session_state.last_context = ""
if "last_distance_info" not in st.session_state:
    st.session_state.last_distance_info = ""

# Function to create the chatbot
def create_chatbot(url):
    with st.spinner("Scraping website content..."):
        website_content = scrape_website(url)
        if not website_content:
            st.error("Failed to scrape website content. Please check the URL and try again.")
            return False
        
        # Store content length for statistics
        content_length = len(website_content)
        st.session_state.content_stats["raw_content_length"] = content_length
        st.session_state.content_stats["content_source"] = url
    
    with st.spinner("Processing content and generating embeddings..."):
        # Process text into chunks
        chunks = process_text(website_content)
        if not chunks:
            st.error("Failed to process website content.")
            return False
        
        # Collect statistics about chunks
        num_chunks = len(chunks)
        avg_chunk_size = sum(len(chunk) for chunk in chunks) / max(1, num_chunks)
        total_chunk_length = sum(len(chunk) for chunk in chunks)
        
        # Store statistics
        st.session_state.content_stats["num_chunks"] = num_chunks
        st.session_state.content_stats["avg_chunk_size"] = int(avg_chunk_size)
        st.session_state.content_stats["total_chunk_length"] = total_chunk_length
        st.session_state.content_stats["processing_ratio"] = total_chunk_length / max(1, content_length)
        
        # Store website and chunks in database
        website_id = db.get_or_create_website(url, website_content)
        db.store_website_chunks(website_id, chunks)
        
        # Create a new chat session
        chat_session_id = db.create_chat_session(website_id)
        
        # Get embeddings and create FAISS index
        try:
            index = create_faiss_index(chunks)
            st.session_state.index = index
            st.session_state.chunks = chunks
            st.session_state.chatbot_ready = True
            st.session_state.website_url = url
            st.session_state.website_id = website_id
            st.session_state.chat_session_id = chat_session_id
            st.session_state.chat_history = []  # Reset chat history for new session
            
            # Log successful chatbot creation with statistics
            print(f"Chatbot created successfully for {url}:")
            print(f"- Raw content length: {content_length} characters")
            print(f"- Number of chunks: {num_chunks}")
            print(f"- Average chunk size: {avg_chunk_size:.1f} characters")
            print(f"- Total processed content: {total_chunk_length} characters")
            
            return True
        except Exception as e:
            st.error(f"Error creating embeddings: {str(e)}")
            return False

# URL input section
with st.container():
    col1, col2 = st.columns([3, 1])
    with col1:
        url = st.text_input("Enter website URL:", 
                            placeholder="https://example.com",
                            value=st.session_state.website_url)
    with col2:
        create_btn = st.button("Create Chatbot", use_container_width=True)

# Process the URL when the button is clicked
if create_btn and url:
    try:
        # Check if we already have this website in the database
        with st.spinner("Checking database for existing website data..."):
            website_id = db.get_or_create_website(url)
        
            # If the website already exists and has chunks, we can load from database
            existing_chunks = db.get_website_chunks(website_id)
        
        if existing_chunks:
            with st.spinner("Loading website data from database..."):
                # Create a new chat session
                chat_session_id = db.create_chat_session(website_id)
                
                # Process the chunks and create index
                try:
                    index = create_faiss_index(existing_chunks)
                    st.session_state.index = index
                    st.session_state.chunks = existing_chunks
                    st.session_state.chatbot_ready = True
                    st.session_state.website_url = url
                    st.session_state.website_id = website_id
                    st.session_state.chat_session_id = chat_session_id
                    st.session_state.chat_history = []  # Start with empty chat for new session
                    
                    st.success("Chatbot loaded from database! You can now ask questions about the website.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error creating embeddings: {str(e)}")
        else:
            # Website doesn't exist or has no chunks, create from scratch
            success = create_chatbot(url)
            if success:
                st.success("Chatbot created successfully! You can now ask questions about the website.")
                st.rerun()
    except Exception as e:
        st.error(f"Database error: {str(e)}. Please try again later.")

# Chat interface
if st.session_state.chatbot_ready:
    # Add sidebar with chat sessions
    with st.sidebar:
        st.markdown("### Previous Chat Sessions")
        if st.session_state.website_id:
            try:
                chat_sessions = db.get_chat_sessions_for_website(st.session_state.website_id)
                
                # Format datetime to be more readable
                formatted_sessions = []
                for session_id, created_at in chat_sessions:
                    # Format the datetime to show only date and time
                    formatted_date = created_at.strftime("%Y-%m-%d %H:%M")
                    formatted_sessions.append((session_id, formatted_date))
                
                if formatted_sessions:
                    # Create a selectbox to choose a session
                    session_options = ["Current Session"] + [f"Session from {date}" for _, date in formatted_sessions]
                    
                    selected_session = st.selectbox("Select a chat session", session_options)
                    
                    # If a previous session is selected, load its history
                    if selected_session != "Current Session":
                        try:
                            selected_index = session_options.index(selected_session) - 1  # Adjust for "Current Session"
                            selected_session_id = formatted_sessions[selected_index][0]
                            
                            if selected_session_id != st.session_state.chat_session_id:
                                # Load chat history from the selected session
                                st.session_state.chat_history = db.get_chat_history(selected_session_id)
                                st.session_state.chat_session_id = selected_session_id
                                st.rerun()
                        except Exception as e:
                            st.sidebar.error(f"Error loading chat session: {str(e)}")
                else:
                    st.sidebar.info("No previous chat sessions found for this website.")
            except Exception as e:
                st.sidebar.error(f"Error retrieving chat sessions: {str(e)}")
    
    # Main content area
    st.markdown(f"### Chat with your AI assistant about: {st.session_state.website_url}")
    
    # Display chat history
    for i, (role, message) in enumerate(st.session_state.chat_history):
        if role == "user":
            st.markdown(f"**You:** {message}")
        else:
            st.markdown(f"**AI:** {message}")
    
    # Initialize session state for handling query and response generation
    if "current_query" not in st.session_state:
        st.session_state.current_query = ""
    if "response_generated" not in st.session_state:
        st.session_state.response_generated = False
    
    # User input for chat
    user_query = st.text_input("Ask a question about the website:", key="user_query")
    
    # Only update the current query if it's different and response hasn't been generated
    if user_query and user_query != st.session_state.current_query:
        st.session_state.current_query = user_query
        st.session_state.response_generated = False
    
    # Generate response button
    if st.button("Generate Response") and st.session_state.current_query and not st.session_state.response_generated:
        # Add user query to chat history if not already added
        if not st.session_state.chat_history or st.session_state.chat_history[-1][0] != "user" or st.session_state.chat_history[-1][1] != st.session_state.current_query:
            # Add user message to session state
            st.session_state.chat_history.append(("user", st.session_state.current_query))
            # Save user message to database
            db.add_chat_message(st.session_state.chat_session_id, "user", st.session_state.current_query)
        
        # Generate response
        with st.spinner("Generating response..."):
            # Search for relevant chunks
            try:
                query_embedding = get_embeddings(st.session_state.current_query)
                similar_chunks_indices, chunk_distances = search_similar_chunks(
                    st.session_state.index, 
                    query_embedding, 
                    k=7,  # Retrieve more chunks for better context
                    distance_threshold=1.5  # Adjust as needed based on testing
                )
                
                # If no relevant chunks found, provide clear error
                if not similar_chunks_indices:
                    response = "I don't have enough information from the website to answer that question. The content might not be available in the extracted data."
                else:
                    # Prepare context with distance-based weighting
                    # Lower distance = more relevant
                    weighted_chunks = []
                    
                    # Sort chunks by relevance (distance)
                    for i, idx in enumerate(similar_chunks_indices):
                        weighted_chunks.append((st.session_state.chunks[idx], chunk_distances[i]))
                    
                    # Sort by distance (lower is better)
                    weighted_chunks.sort(key=lambda x: x[1])
                    
                    # Extract just the chunks in order of relevance
                    context_chunks = [chunk for chunk, _ in weighted_chunks]
                    
                    # Create a format that highlights the most relevant chunks first
                    context = "\n\n---\n\n".join(context_chunks)
                    
                    # Store the context for debugging purposes
                    st.session_state.last_context = context
                    
                    # Store distance info for debugging
                    if st.session_state.debug_mode:
                        distance_info = []
                        for i, (chunk, dist) in enumerate(weighted_chunks):
                            # Only include first 100 chars of each chunk for brevity
                            preview = chunk[:100] + "..." if len(chunk) > 100 else chunk
                            distance_info.append(f"Chunk {i+1} (distance: {dist:.4f}): {preview}")
                        st.session_state.last_distance_info = "\n\n".join(distance_info)
                
                    # Generate response using OpenAI
                    response = generate_response(st.session_state.current_query, context, st.session_state.website_url)
                
                # Add response to chat history in session state
                st.session_state.chat_history.append(("ai", response))
                
                # Save AI response to database
                db.add_chat_message(st.session_state.chat_session_id, "ai", response)
                
                # Mark response as generated to prevent multiple generations
                st.session_state.response_generated = True
                
                st.rerun()
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")
    
    # Add debug mode toggle and content statistics
    with st.expander("Advanced Options"):
        debug_toggle = st.checkbox("Enable Debug Mode", value=st.session_state.debug_mode)
        if debug_toggle != st.session_state.debug_mode:
            st.session_state.debug_mode = debug_toggle
            st.rerun()
        
        if st.session_state.debug_mode:
            st.markdown("### Content Statistics")
            
            # Display detailed content statistics from session state
            if st.session_state.content_stats:
                # Basic stats
                st.markdown("#### Website Content")
                st.markdown(f"**Source URL:** {st.session_state.content_stats.get('content_source', 'N/A')}")
                st.markdown(f"**Raw content length:** {st.session_state.content_stats.get('raw_content_length', 0):,} characters")
                
                # Chunk stats
                st.markdown("#### Chunking Statistics")
                st.markdown(f"**Total chunks:** {st.session_state.content_stats.get('num_chunks', len(st.session_state.chunks))}")
                st.markdown(f"**Average chunk size:** {st.session_state.content_stats.get('avg_chunk_size', 0):,} characters")
                st.markdown(f"**Processed content length:** {st.session_state.content_stats.get('total_chunk_length', 0):,} characters")
                
                # Processing ratio (how much of the original content was kept after processing)
                if 'processing_ratio' in st.session_state.content_stats:
                    ratio = st.session_state.content_stats['processing_ratio'] * 100
                    st.markdown(f"**Processing efficiency:** {ratio:.1f}% of content preserved")
            else:
                # If no stats yet, just show basic info
                st.markdown(f"**Total chunks:** {len(st.session_state.chunks)}")
                
                # Calculate average chunk length
                if st.session_state.chunks:
                    avg_length = sum(len(chunk) for chunk in st.session_state.chunks) / len(st.session_state.chunks)
                    st.markdown(f"**Average chunk length:** {avg_length:.0f} characters")
            
            # Query information
            if hasattr(st.session_state, 'last_query_info') and st.session_state.last_query_info:
                st.markdown("#### Last Query Debug")
                st.markdown(f"**Query:** {st.session_state.last_query_info.get('query', 'N/A')}")
                st.markdown(f"**Chunks found:** {st.session_state.last_query_info.get('chunks_found', 0)}")
            
            # Display sample chunks
            st.markdown("#### Content Inspection")
            if st.button("View Sample Content"):
                # Show a sample of chunks (up to 5)
                st.markdown("##### Sample Content Chunks")
                sample_size = min(5, len(st.session_state.chunks))
                for i in range(sample_size):
                    with st.expander(f"Chunk {i+1}"):
                        st.text(st.session_state.chunks[i][:500] + "..." if len(st.session_state.chunks[i]) > 500 else st.session_state.chunks[i])
            
            # Add buttons to view debug information
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("View Last Query Context"):
                    if hasattr(st.session_state, 'last_context') and st.session_state.last_context:
                        st.markdown("##### Context Used for Last Query")
                        st.text_area("Context:", st.session_state.last_context, height=300)
                    else:
                        st.info("No context available yet. Ask a question first.")
            
            with col2:
                if st.button("View Similarity Scores"):
                    if hasattr(st.session_state, 'last_distance_info') and st.session_state.last_distance_info:
                        st.markdown("##### Similarity Information")
                        st.markdown("Lower distance values indicate higher relevance:")
                        st.text_area("Distance Scores:", st.session_state.last_distance_info, height=300)
                    else:
                        st.info("No similarity data available yet. Ask a question first.")
    
    # Reset chatbot button
    if st.button("Create new chatbot"):
        # Reset main state variables
        st.session_state.chatbot_ready = False
        st.session_state.index = None
        st.session_state.chunks = []
        st.session_state.chat_history = []
        st.session_state.website_url = ""
        st.session_state.website_id = None
        st.session_state.chat_session_id = None
        
        # Reset query and response tracking
        st.session_state.current_query = ""
        st.session_state.response_generated = False
        
        # Reset all debug information
        st.session_state.content_stats = {}
        st.session_state.last_query_info = {}
        st.session_state.last_context = ""
        st.session_state.last_distance_info = ""
        
        st.rerun()
