import os
import trafilatura
import requests
from bs4 import BeautifulSoup
import json

def get_website_text_content(url: str) -> str:
    """
    Extract the main text content of a website using trafilatura.
    
    Args:
        url (str): The URL of the website to extract content from.
        
    Returns:
        str: The extracted text content.
    """
    try:
        downloaded = trafilatura.fetch_url(url)
        text = trafilatura.extract(downloaded)
        return text
    except Exception as e:
        print(f"Error extracting content with trafilatura: {str(e)}")
        return None

def scrape_with_serpapi(url: str) -> str:
    """
    Scrape website content using SerpAPI.
    
    Args:
        url (str): The URL of the website to scrape.
        
    Returns:
        str: The extracted content.
    """
    try:
        # Get SerpAPI key from environment variable
        serpapi_key = os.getenv("SERPAPI_KEY")
        if not serpapi_key:
            print("SerpAPI key not found in environment variables.")
            return None
            
        # Prepare the request to SerpAPI
        params = {
            "engine": "google",
            "q": f"site:{url}",
            "api_key": serpapi_key,
            "num": 10  # Number of results to retrieve
        }
        
        response = requests.get("https://serpapi.com/search", params=params)
        data = response.json()
        
        # Extract content from the search results
        if "organic_results" in data:
            content_parts = []
            for result in data["organic_results"]:
                if "snippet" in result:
                    content_parts.append(result["snippet"])
                    
            return "\n\n".join(content_parts)
        else:
            print("No organic results found in SerpAPI response.")
            return None
            
    except Exception as e:
        print(f"Error scraping with SerpAPI: {str(e)}")
        return None

def scrape_website(url: str) -> str:
    """
    Scrape website content using available methods.
    First tries SerpAPI, and falls back to trafilatura if needed.
    
    Args:
        url (str): The URL of the website to scrape.
        
    Returns:
        str: The extracted content.
    """
    # First try using SerpAPI
    serpapi_content = scrape_with_serpapi(url)
    if serpapi_content and len(serpapi_content) > 100:  # Ensure we got meaningful content
        print("Successfully scraped content using SerpAPI")
        return serpapi_content
    
    # Fallback to trafilatura
    print("Falling back to trafilatura for content extraction")
    trafilatura_content = get_website_text_content(url)
    if trafilatura_content:
        return trafilatura_content
    
    # If both methods fail, return error
    print("Failed to scrape website content with all available methods")
    return None
