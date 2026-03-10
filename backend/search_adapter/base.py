from abc import ABC, abstractmethod
from typing import List, Dict, Any


class SearchAdapter(ABC):
    """Abstract base class for search adapters."""

    @abstractmethod
    def search(self, query: str, k: int = 5) -> Dict[str, Any]:
        """
        Execute a search query and return results and an optional summary answer.

        Args:
            query: The search query string.
            k: The maximum number of results to return.

        Returns:
            A dictionary containing:
            - results: A list of result dictionaries (title, snippet, url, source, raw).
            - answer: An optional high-level summary answer if provided by the adapter.
        """
        pass
