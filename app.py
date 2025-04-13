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
                similar_chunks_indices = search_similar_chunks(
                    st.session_state.index, 
                    query_embedding, 
                    k=3
                )
                
                context_chunks = [st.session_state.chunks[i] for i in similar_chunks_indices]
                context = "\n\n".join(context_chunks)
                
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
    
    # Reset chatbot button
    if st.button("Create new chatbot"):
        st.session_state.chatbot_ready = False
        st.session_state.index = None
        st.session_state.chunks = []
        st.session_state.chat_history = []
        st.session_state.website_url = ""
        st.session_state.website_id = None
        st.session_state.chat_session_id = None
        st.rerun()
