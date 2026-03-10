import re
import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


class FeatureParser:
    """Parser to extract structured features from search results content."""

    def __init__(self):
        # Regex patterns for common football result formats
        self.score_pattern = re.compile(r"(\d+)\s*[-–]\s*(\d+)")
        self.result_pattern = re.compile(r"\b(won|beat|lost|defeated|drew|draw)\b", re.IGNORECASE)

    def extract_snippet_features(self, results_by_query: Dict[str, Dict[str, Any]], team_a: str, team_b: str) -> Dict[str, Any]:
        """
        Extract features from a collection of search results.

        Args:
            results_by_query: Mapping of query to search response (results and answer).
            team_a: Name of Team A.
            team_b: Name of Team B.

        Returns:
            A dictionary of extracted features.
        """
        features = {
            "teamA": {"name": team_a, "last_5": [], "rank": "N/A"},
            "teamB": {"name": team_b, "last_5": [], "rank": "N/A"},
            "h2h": {"summary": "N/A", "recent_results": []},
            "league": "N/A",
            "search_summaries": [] # New field for Tavily answers
        }

        for query, response in results_by_query.items():
            query_lower = query.lower()
            results = response.get("results", [])
            answer = response.get("answer", "")
            
            if answer:
                features["search_summaries"].append(f"Query: {query}\nAnswer: {answer}")
                # Also run heuristics on the answer itself
                self._extract_rank(answer, team_a, team_b, features)
                self._extract_match_results(answer, team_a, team_b, features)

            for res in results:
                content = res.get("snippet", "")
                
                # Try to extract ranks if query refers to table or ranking
                if "rank" in query_lower or "table" in query_lower or "standing" in query_lower:
                    self._extract_rank(content, team_a, team_b, features)

                # Try to extract match results
                self._extract_match_results(content, team_a, team_b, features)

        # Post-process results to be concise
        features["teamA"]["last_5"] = features["teamA"]["last_5"][:5]
        features["teamB"]["last_5"] = features["teamB"]["last_5"][:5]
        
        return features

    def _extract_rank(self, text: str, team_a: str, team_b: str, features: Dict[str, Any]):
        """Helper to extract league rank from text."""
        # Heuristic: look for "Team X is 4th", "4th Team X", "Team X ... 4th"
        for team, key in [(team_a, "teamA"), (team_b, "teamB")]:
            # Pattern 1: 1st/2nd/3rd etc then team name
            pattern1 = rf"(\d+)(?:st|nd|rd|th)?\s+(?:in|position|place|ranked|of|the)?\s+{re.escape(team)}"
            # Pattern 2: team name then is 1st/2nd/3rd etc
            pattern2 = rf"{re.escape(team)}\s+(?:is|are|at|currently|ranked|position|in)?\s+(?:in\s+)?(?:\w+\s+)?(\d+)(?:st|nd|rd|th)?"
            
            match = re.search(pattern1, text, re.IGNORECASE)
            if not match:
                match = re.search(pattern2, text, re.IGNORECASE)
            
            if match:
                features[key]["rank"] = match.group(1)

    def _extract_match_results(self, text: str, team_a: str, team_b: str, features: Dict[str, Any]):
        """Helper to extract match results (W/D/L) from text."""
        # Split text into potential sentences/blocks to avoid one-line-captures-all
        blocks = re.split(r'[.!?\n]', text)
        
        for block in blocks:
            block = block.strip()
            if not block:
                continue

            # Check for head-to-head first
            if team_a.lower() in block.lower() and team_b.lower() in block.lower():
                scores = self.score_pattern.findall(block)
                if scores or self.result_pattern.search(block):
                    if block not in features["h2h"]["recent_results"]:
                        features["h2h"]["recent_results"].append(block)
            
            # Check for individual team results
            for team, key in [(team_a, "teamA"), (team_b, "teamB")]:
                if team.lower() in block.lower():
                    # Look for scoreline or result verbs
                    scores = self.score_pattern.findall(block)
                    if scores or self.result_pattern.search(block):
                        if block not in features[key]["last_5"]:
                            features[key]["last_5"].append(block)

    def fetch_extracted_content_fallback(self, adapter: Any, urls: List[str]) -> str:
        """
        Fallback to fetch full extracted content for a list of URLs.

        Args:
            adapter: The TavilyAdapter instance.
            urls: List of URLs.

        Returns:
            Concatenated extracted text.
        """
        try:
            extraction = adapter.extract(urls[:2])  # Limit to top 2 for efficiency
            texts = [res.get("raw_content", "") for res in extraction.get("results", [])]
            return "\n".join(texts)
        except Exception as e:
            logger.warning(f"Feature extraction fallback failed: {e}")
            return ""
