import os
import json
import time
import logging
from datetime import datetime
from typing import Dict, Any, List

from .search_adapter.tavily import TavilyAdapter
from .llm.llm_client import LocalLLM
from .parse.parser import FeatureParser

logger = logging.getLogger(__name__)

# Authoritative prompts from plan.md
QUERY_GEN_PROMPT_TEMPLATE = """You are a concise query generator.

Task:
Given the request: "{request}"

Output EXACTLY a JSON array (no commentary), containing 3 to 6 short Google-style search queries that retrieve:
last 5 match results for team A,
last 5 match results for team B,
head-to-head matches,
league ranking pages.

Rules:
Each query <= 8 words.
Do not add explanations or text outside the JSON array.

Example output:
["Arsenal last 5 matches results", "Arsenal fixtures last 5", "Arsenal vs Chelsea head-to-head"]
"""

PREDICTION_PROMPT_TEMPLATE = """You are an objective football analyst. Given the structured summary below, answer in strict JSON with fields:
{{"winner": "<Team A|Team B|Draw>", "probability": <0-100 number>, "reasoning": "<one-sentence reasoning>", "top_signals": ["signal1","signal2","signal3"]}}

Data:
Team A: {team_a}
Team B: {team_b}
Extracted features JSON:
{features}

Rules:
Output only a single JSON object.
Probability is a number between 0 and 100 that reflects confidence.
Provide three concise top_signals supporting the decision.
"""

def run_prediction(team_a: str, team_b: str) -> Dict[str, Any]:
    """
    Orchestrate the end-to-end prediction flow.

    Args:
        team_a: Name of the first team.
        team_b: Name of the second team.

    Returns:
        The prediction JSON object.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Initialize components
    search_adapter = TavilyAdapter()
    llm = LocalLLM()
    parser = FeatureParser()

    # 2. Query Generation
    request_text = f"Analyze the match between {team_a} and {team_b} including recent form, H2H, and standings."
    query_prompt = QUERY_GEN_PROMPT_TEMPLATE.format(request=request_text)
    queries = llm.generate_json(query_prompt)
    if not isinstance(queries, list):
        logger.error(f"LLM did not return a list of queries: {queries}")
        queries = [f"{team_a} vs {team_b} results", f"{team_a} last matches", f"{team_b} last matches"]

    # 3. Search Data Collection
    raw_results = {}
    for query in queries:
        logger.info(f"Searching for: {query}")
        results = search_adapter.search(query, k=5)
        raw_results[query] = results
        time.sleep(0.5)  # Simple rate limit

    # 4. Feature Extraction
    features = parser.extract_snippet_features(raw_results, team_a, team_b)
    
    # 5. Check for sufficient signals and fallback to Extract if needed
    # Rule: fewer than 3 explicit outcomes per team (simplification for MVP)
    if len(features["teamA"]["last_5"]) < 3 or len(features["teamB"]["last_5"]) < 3:
        logger.info("Insufficient signals in snippets, falling back to Tavily Extract.")
        all_urls = []
        for response in raw_results.values():
            all_urls.extend([res["url"] for res in response.get("results", [])])
        
        if all_urls:
            extra_content = parser.fetch_extracted_content_fallback(search_adapter, all_urls)
            # Re-run extraction with extra content (pseudo-query for storage)
            raw_results["TAVILY_EXTRACT_FALLBACK"] = {
                "results": [{"snippet": extra_content, "url": "fallback"}],
                "answer": ""
            }
            features = parser.extract_snippet_features(raw_results, team_a, team_b)

    # 6. Prediction
    prediction_prompt = PREDICTION_PROMPT_TEMPLATE.format(
        team_a=team_a,
        team_b=team_b,
        features=json.dumps(features, indent=2)
    )
    prediction = llm.generate_json(prediction_prompt)

    # 7. Logging
    log_data = {
        "teams": {"team_a": team_a, "team_b": team_b},
        "timestamp": timestamp,
        "queries": queries,
        "raw_tavily": raw_results,
        "features": features,
        "prompts_used": {
            "query_gen": query_prompt,
            "prediction": prediction_prompt
        },
        "llm_responses": {
            "queries": queries,
            "prediction": prediction
        },
        "parsed_prediction": prediction
    }
    
    os.makedirs("runs", exist_ok=True)
    log_path = f"runs/run_{timestamp}.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2)

    prediction["log_path"] = log_path
    return prediction
