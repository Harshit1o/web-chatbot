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

# Function to create the chatbot
def create_chatbot(url):
    with st.spinner("Scraping website content..."):
        website_content = scrape_website(url)
        if not website_content:
            st.error("Failed to scrape website content. Please check the URL and try again.")
            return False
    
    with st.spinner("Processing content and generating embeddings..."):
        chunks = process_text(website_content)
        if not chunks:
            st.error("Failed to process website content.")
            return False
        
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
    success = create_chatbot(url)
    if success:
        st.success("Chatbot created successfully! You can now ask questions about the website.")
        st.rerun()

# Chat interface
if st.session_state.chatbot_ready:
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
            st.session_state.chat_history.append(("user", st.session_state.current_query))
        
        # Generate response
        with st.spinner("Generating response..."):
            # Search for relevant chunks
            try:
                query_embedding = get_embeddings(st.session_state.current_query)
                similar_chunks_indices = search_similar_chunks(
                    st.session_state.index, 
                    query_embedding, 
                    k=3
                )
                
                context_chunks = [st.session_state.chunks[i] for i in similar_chunks_indices]
                context = "\n\n".join(context_chunks)
                
                # Generate response using OpenAI
                response = generate_response(st.session_state.current_query, context, st.session_state.website_url)
                
                # Add response to chat history
                st.session_state.chat_history.append(("ai", response))
                
                # Mark response as generated to prevent multiple generations
                st.session_state.response_generated = True
                
                st.rerun()
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")
    
    # Reset chatbot button
    if st.button("Create new chatbot"):
        st.session_state.chatbot_ready = False
        st.session_state.index = None
        st.session_state.chunks = []
        st.session_state.chat_history = []
        st.session_state.website_url = ""
        st.rerun()
