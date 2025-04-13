import os
from sqlalchemy import Column, Integer, String, Text, DateTime, create_engine, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import datetime

# Create database connection
db_url = os.environ.get("DATABASE_URL")
engine = create_engine(db_url)
Session = sessionmaker(bind=engine)
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
    Base.metadata.create_all(engine)

# Database operations
def get_or_create_website(url, content=None):
    """
    Get a website by URL or create it if it doesn't exist
    """
    session = Session()
    website = session.query(Website).filter(Website.url == url).first()
    
    if not website:
        website = Website(url=url, content=content)
        session.add(website)
        session.commit()
    
    website_id = website.id
    session.close()
    return website_id

def store_website_chunks(website_id, chunks):
    """
    Store content chunks for a website
    """
    session = Session()
    
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
    session.close()

def get_website_chunks(website_id):
    """
    Get all content chunks for a website
    """
    session = Session()
    chunks = session.query(ContentChunk).filter(
        ContentChunk.website_id == website_id
    ).order_by(ContentChunk.chunk_index).all()
    
    chunk_texts = [chunk.chunk_text for chunk in chunks]
    session.close()
    return chunk_texts

def create_chat_session(website_id):
    """
    Create a new chat session for a website
    """
    session = Session()
    chat_session = ChatSession(website_id=website_id)
    session.add(chat_session)
    session.commit()
    
    session_id = chat_session.id
    session.close()
    return session_id

def add_chat_message(session_id, role, content):
    """
    Add a message to a chat session
    """
    session = Session()
    message = ChatMessage(
        session_id=session_id,
        role=role,
        content=content
    )
    session.add(message)
    session.commit()
    session.close()

def get_chat_history(session_id):
    """
    Get all messages in a chat session
    """
    session = Session()
    messages = session.query(ChatMessage).filter(
        ChatMessage.session_id == session_id
    ).order_by(ChatMessage.created_at).all()
    
    history = [(msg.role, msg.content) for msg in messages]
    session.close()
    return history

def get_chat_sessions_for_website(website_id):
    """
    Get all chat sessions for a website
    """
    session = Session()
    chat_sessions = session.query(ChatSession).filter(
        ChatSession.website_id == website_id
    ).order_by(ChatSession.created_at.desc()).all()
    
    sessions = [(chat.id, chat.created_at) for chat in chat_sessions]
    session.close()
    return sessions

# Initialize the database on import
init_db()