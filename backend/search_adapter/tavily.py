import os
import time
import logging
from typing import List, Dict, Any
from tavily import TavilyClient
from dotenv import load_dotenv

from .base import SearchAdapter

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class TavilyAdapter(SearchAdapter):
    """Tavily implementation of the SearchAdapter."""

    def __init__(self, api_key: str | None = None):
        """
        Initialize the TavilyAdapter.

        Args:
            api_key: The Tavily API key. If not provided, it will be read from TAVILY_API_KEY env var.
        """
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        if not self.api_key:
            raise ValueError("TAVILY_API_KEY not found in environment or arguments.")
        self.client = TavilyClient(api_key=self.api_key)

    def search(self, query: str, k: int = 5, retry_count: int = 3) -> Dict[str, Any]:
        """
        Execute a search query using Tavily.

        Args:
            query: The search query string.
            k: The maximum number of results to return.
            retry_count: Number of retries for transient errors.

        Returns:
            A dictionary containing "results" list and an optional "answer".
        """
        for attempt in range(retry_count):
            try:
                # include_answer="advanced" provides a clean 'answer' field
                response = self.client.search(
                    query=query, 
                    max_results=k, 
                    include_answer="advanced",
                    search_depth="advanced"
                )
                
                results = []
                for res in response.get("results", []):
                    results.append({
                        "title": res.get("title", ""),
                        "snippet": res.get("content", ""),
                        "url": res.get("url", ""),
                        "source": "tavily",
                        "raw": res
                    })
                
                return {
                    "results": results,
                    "answer": response.get("answer", "")
                }

            except Exception as e:
                logger.warning(f"Tavily search attempt {attempt + 1} failed: {e}")
                if attempt < retry_count - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"Tavily search failed after {retry_count} attempts.")
                    raise

        return []

    def extract(self, urls: List[str]) -> Dict[str, Any]:
        """
        Call Tavily Extract for a list of URLs.

        Args:
            urls: List of URLs to extract content from.

        Returns:
            The raw response from Tavily Extract.
        """
        try:
            return self.client.extract(urls=urls)
        except Exception as e:
            logger.error(f"Tavily extract failed: {e}")
            return {}
