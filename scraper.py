import os
import trafilatura
import requests
from bs4 import BeautifulSoup
import json
import re
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_website_text_content(url: str) -> str:
    """
    Extract the main text content of a website using trafilatura.
    
    Args:
        url (str): The URL of the website to extract content from.
        
    Returns:
        str: The extracted text content.
    """
    try:
        # First try with default settings
        downloaded = trafilatura.fetch_url(url)
        text = trafilatura.extract(downloaded, include_comments=False, include_tables=True, 
                                  include_links=True, include_images=False)
        
        # If the content is too short, try with alternative settings
        if not text or len(text) < 500:
            logger.info("Initial extraction returned limited content, trying with alternative settings")
            text = trafilatura.extract(downloaded, favor_precision=False, favor_recall=True, 
                                      include_comments=False, include_tables=True, 
                                      include_links=True, include_images=False)
        
        # If still no good content, try with BeautifulSoup as a fallback
        if not text or len(text) < 500:
            logger.info("Trafilatura extraction failed, trying with BeautifulSoup")
            return extract_with_beautifulsoup(url)
            
        return text
    except Exception as e:
        logger.error(f"Error extracting content with trafilatura: {str(e)}")
        # Try with BeautifulSoup as a fallback
        return extract_with_beautifulsoup(url)

def extract_with_beautifulsoup(url: str) -> str:
    """
    Extract content using BeautifulSoup as a fallback method.
    
    Args:
        url (str): The URL of the website to extract content from.
        
    Returns:
        str: The extracted text content.
    """
    try:
        response = requests.get(url, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script_or_style in soup(['script', 'style', 'meta', 'noscript', 'header', 'footer', 'nav']):
            script_or_style.decompose()
        
        # Extract text from p, h1-h6, li, and div elements
        paragraphs = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li'])
        
        # If few paragraphs found, try to get text from divs with substantial content
        if len(paragraphs) < 10:
            divs = soup.find_all('div')
            for div in divs:
                div_text = div.get_text(strip=True)
                if len(div_text) > 100:  # Only include divs with substantial content
                    paragraphs.append(div)
        
        # Extract text and clean it
        content_parts = []
        for elem in paragraphs:
            text = elem.get_text(strip=True)
            if text and len(text) > 20:  # Minimum length to avoid noise
                content_parts.append(text)
        
        content = "\n\n".join(content_parts)
        
        # Clean up the content
        content = re.sub(r'\s+', ' ', content)  # Replace multiple whitespace with single space
        content = re.sub(r'\n\s*\n', '\n\n', content)  # Replace multiple newlines
        
        return content
    except Exception as e:
        logger.error(f"Error extracting content with BeautifulSoup: {str(e)}")
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
            logger.warning("SerpAPI key not found in environment variables.")
            return None
            
        # Parse the URL to get the domain
        domain = url.split('//')[1].split('/')[0] if '//' in url else url.split('/')[0]
        
        # Prepare the request to SerpAPI - query for the specific URL first
        params = {
            "engine": "google",
            "q": f"site:{domain}",
            "api_key": serpapi_key,
            "num": 25  # Increased number of results to retrieve
        }
        
        logger.info(f"Querying SerpAPI for domain: {domain}")
        response = requests.get("https://serpapi.com/search", params=params)
        data = response.json()
        
        # Extract content from the search results
        if "organic_results" in data:
            content_parts = []
            
            # Extract snippets and titles
            for result in data["organic_results"]:
                # Add title if available
                if "title" in result:
                    content_parts.append(f"TITLE: {result['title']}")
                
                # Add snippet if available
                if "snippet" in result:
                    content_parts.append(f"CONTENT: {result['snippet']}")
                
                # Add a separator between results
                content_parts.append("---")
                    
            return "\n".join(content_parts)
        else:
            logger.warning("No organic results found in SerpAPI response.")
            return None
            
    except Exception as e:
        logger.error(f"Error scraping with SerpAPI: {str(e)}")
        return None

def scrape_website(url: str) -> str:
    """
    Scrape website content using multiple methods and combine the results
    for comprehensive content collection.
    
    Args:
        url (str): The URL of the website to scrape.
        
    Returns:
        str: The combined extracted content.
    """
    all_content = []
    
    # Try direct extraction with trafilatura/BeautifulSoup
    logger.info(f"Extracting content directly from: {url}")
    direct_content = get_website_text_content(url)
    if direct_content and len(direct_content) > 200:
        logger.info(f"Successfully extracted {len(direct_content)} characters directly from the URL")
        all_content.append("DIRECT CONTENT:\n" + direct_content)
    
    # Try using SerpAPI
    logger.info("Extracting content using SerpAPI")
    serpapi_content = scrape_with_serpapi(url)
    if serpapi_content and len(serpapi_content) > 200:
        logger.info(f"Successfully extracted {len(serpapi_content)} characters using SerpAPI")
        all_content.append("SEARCH RESULTS CONTENT:\n" + serpapi_content)
    
    # Combine all content sources
    if all_content:
        combined_content = "\n\n" + "\n\n==============\n\n".join(all_content)
        logger.info(f"Combined content length: {len(combined_content)} characters")
        return combined_content
    
    # If all methods fail, return error
    logger.error("Failed to scrape website content with all available methods")
    return None
