import pytest
from backend.parse.parser import FeatureParser

def test_extract_rank():
    parser = FeatureParser()
    features = {
        "teamA": {"rank": "N/A"},
        "teamB": {"rank": "N/A"}
    }
    
    parser._extract_rank("Arsenal is currently 1st in the league", "Arsenal", "Chelsea", features)
    assert features["teamA"]["rank"] == "1"
    
    parser._extract_rank("Chelsea are 4th after the last win", "Arsenal", "Chelsea", features)
    assert features["teamB"]["rank"] == "4"

def test_extract_match_results():
    parser = FeatureParser()
    features = {
        "teamA": {"last_5": []},
        "teamB": {"last_5": []},
        "h2h": {"recent_results": []}
    }
    
    text = "Arsenal beat Chelsea 2-0 yesterday. Arsenal won against Wolves 1-0."
    parser._extract_match_results(text, "Arsenal", "Chelsea", features)
    
    assert len(features["teamA"]["last_5"]) >= 2
    assert len(features["h2h"]["recent_results"]) >= 1
    assert "2-0" in features["teamA"]["last_5"][0]

def test_extract_snippet_features():
    parser = FeatureParser()
    results_by_query = {
        "Arsenal rank": {
            "results": [{"snippet": "Arsenal are 1st"}],
            "answer": "Arsenal are currently at the top of the table."
        },
        "Arsenal last 5": {
            "results": [{"snippet": "Arsenal 2-0 Everton"}],
            "answer": "Arsenal beat Everton 2-0."
        }
    }
    
    features = parser.extract_snippet_features(results_by_query, "Arsenal", "Everton")
    
    assert features["teamA"]["rank"] == "1"
    assert len(features["teamA"]["last_5"]) >= 1
    assert len(features["search_summaries"]) == 2
