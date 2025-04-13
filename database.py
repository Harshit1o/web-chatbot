import os
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from sqlalchemy import Column, Integer, String, Text, DateTime, create_engine, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, scoped_session
from sqlalchemy.exc import OperationalError, SQLAlchemyError
import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create database connection with retry logic
def get_engine():
    db_url = os.environ.get("DATABASE_URL")
    max_retries = 3
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            engine = create_engine(db_url, pool_pre_ping=True, pool_recycle=3600)
            # Test the connection
            connection = engine.connect()
            connection.close()
            logger.info("Database connection established successfully")
            return engine
        except OperationalError as e:
            logger.warning(f"Database connection attempt {attempt+1}/{max_retries} failed: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                logger.error("Failed to connect to database after multiple attempts")
                raise

engine = get_engine()
session_factory = sessionmaker(bind=engine)
Session = scoped_session(session_factory)
Base = declarative_base()

class Website(Base):
    """
    Model to store website information and content
    """
    __tablename__ = 'websites'
    
    id = Column(Integer, primary_key=True)
    url = Column(String(500), unique=True, nullable=False)
    content = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    # Relationship with chunks
    chunks = relationship("ContentChunk", back_populates="website", cascade="all, delete-orphan")
    # Relationship with chat sessions
    chat_sessions = relationship("ChatSession", back_populates="website", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Website(id={self.id}, url='{self.url}')>"

class ContentChunk(Base):
    """
    Model to store content chunks for a website
    """
    __tablename__ = 'content_chunks'
    
    id = Column(Integer, primary_key=True)
    website_id = Column(Integer, ForeignKey('websites.id'), nullable=False)
    chunk_text = Column(Text, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    
    # Relationship with website
    website = relationship("Website", back_populates="chunks")
    
    def __repr__(self):
        return f"<ContentChunk(id={self.id}, website_id={self.website_id}, chunk_index={self.chunk_index})>"

class ChatSession(Base):
    """
    Model to store chat sessions
    """
    __tablename__ = 'chat_sessions'
    
    id = Column(Integer, primary_key=True)
    website_id = Column(Integer, ForeignKey('websites.id'), nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    # Relationship with website
    website = relationship("Website", back_populates="chat_sessions")
    # Relationship with messages
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<ChatSession(id={self.id}, website_id={self.website_id})>"

class ChatMessage(Base):
    """
    Model to store chat messages
    """
    __tablename__ = 'chat_messages'
    
    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey('chat_sessions.id'), nullable=False)
    role = Column(String(20), nullable=False)  # 'user' or 'ai'
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    # Relationship with session
    session = relationship("ChatSession", back_populates="messages")
    
    def __repr__(self):
        return f"<ChatMessage(id={self.id}, session_id={self.session_id}, role='{self.role}')>"

# Create database tables
def init_db():
    try:
        Base.metadata.create_all(engine)
        logger.info("Database tables created successfully")
    except SQLAlchemyError as e:
        logger.error(f"Error creating database tables: {str(e)}")
        # Don't raise here to allow the application to start even if tables can't be created initially
        # They might be created later when the connection is stable

# Database operations
def get_or_create_website(url, content=None):
    """
    Get a website by URL or create it if it doesn't exist
    """
    session = Session()
    try:
        website = session.query(Website).filter(Website.url == url).first()
        
        if not website:
            website = Website(url=url, content=content)
            session.add(website)
            session.commit()
            logger.info(f"Created new website record for URL: {url}")
        else:
            logger.info(f"Found existing website record for URL: {url}")
        
        website_id = website.id
        return website_id
    except SQLAlchemyError as e:
        session.rollback()
        logger.error(f"Database error in get_or_create_website: {str(e)}")
        raise
    finally:
        session.close()

def store_website_chunks(website_id, chunks):
    """
    Store content chunks for a website
    """
    session = Session()
    try:
        # Delete existing chunks for this website
        session.query(ContentChunk).filter(ContentChunk.website_id == website_id).delete()
        
        # Add new chunks
        for i, chunk_text in enumerate(chunks):
            chunk = ContentChunk(
                website_id=website_id,
                chunk_text=chunk_text,
                chunk_index=i
            )
            session.add(chunk)
        
        session.commit()
        logger.info(f"Stored {len(chunks)} chunks for website_id: {website_id}")
    except SQLAlchemyError as e:
        session.rollback()
        logger.error(f"Database error in store_website_chunks: {str(e)}")
        raise
    finally:
        session.close()

def get_website_chunks(website_id):
    """
    Get all content chunks for a website
    """
    session = Session()
    try:
        chunks = session.query(ContentChunk).filter(
            ContentChunk.website_id == website_id
        ).order_by(ContentChunk.chunk_index).all()
        
        chunk_texts = [chunk.chunk_text for chunk in chunks]
        logger.info(f"Retrieved {len(chunk_texts)} chunks for website_id: {website_id}")
        return chunk_texts
    except SQLAlchemyError as e:
        logger.error(f"Database error in get_website_chunks: {str(e)}")
        return []
    finally:
        session.close()

def create_chat_session(website_id):
    """
    Create a new chat session for a website
    """
    session = Session()
    try:
        chat_session = ChatSession(website_id=website_id)
        session.add(chat_session)
        session.commit()
        
        session_id = chat_session.id
        logger.info(f"Created new chat session {session_id} for website_id: {website_id}")
        return session_id
    except SQLAlchemyError as e:
        session.rollback()
        logger.error(f"Database error in create_chat_session: {str(e)}")
        raise
    finally:
        session.close()

def add_chat_message(session_id, role, content):
    """
    Add a message to a chat session
    """
    session = Session()
    try:
        message = ChatMessage(
            session_id=session_id,
            role=role,
            content=content
        )
        session.add(message)
        session.commit()
        logger.info(f"Added {role} message to session_id: {session_id}")
    except SQLAlchemyError as e:
        session.rollback()
        logger.error(f"Database error in add_chat_message: {str(e)}")
        # Don't raise here to prevent breaking the chat flow if message storage fails
    finally:
        session.close()

def get_chat_history(session_id):
    """
    Get all messages in a chat session
    """
    session = Session()
    try:
        messages = session.query(ChatMessage).filter(
            ChatMessage.session_id == session_id
        ).order_by(ChatMessage.created_at).all()
        
        history = [(msg.role, msg.content) for msg in messages]
        logger.info(f"Retrieved {len(history)} messages for session_id: {session_id}")
        return history
    except SQLAlchemyError as e:
        logger.error(f"Database error in get_chat_history: {str(e)}")
        return []
    finally:
        session.close()

def get_chat_sessions_for_website(website_id):
    """
    Get all chat sessions for a website
    """
    session = Session()
    try:
        chat_sessions = session.query(ChatSession).filter(
            ChatSession.website_id == website_id
        ).order_by(ChatSession.created_at.desc()).all()
        
        sessions = [(chat.id, chat.created_at) for chat in chat_sessions]
        logger.info(f"Retrieved {len(sessions)} chat sessions for website_id: {website_id}")
        return sessions
    except SQLAlchemyError as e:
        logger.error(f"Database error in get_chat_sessions_for_website: {str(e)}")
        return []
    finally:
        session.close()

# Initialize the database on import
init_db()