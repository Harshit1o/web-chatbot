import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# Load environment variables
load_dotenv()

def test_database_connection():
    try:
        # Get the database URL from environment variables
        database_url = os.getenv('DATABASE_URL')
        
        if not database_url:
            return "Error: DATABASE_URL environment variable not found"
        
        print(f"Attempting to connect with: {database_url}")
        
        # Create an engine
        engine = create_engine(database_url)
        
        # Test the connection
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            return "Database connection successful!"
    
    except Exception as e:
        return f"Database connection failed: {str(e)}"

if __name__ == "__main__":
    print(test_database_connection())