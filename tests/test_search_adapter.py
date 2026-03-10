import pytest
from unittest.mock import MagicMock, patch
from backend.search_adapter.tavily import TavilyAdapter

@pytest.fixture
def mock_tavily_client():
    with patch("backend.search_adapter.tavily.TavilyClient") as mock:
        yield mock

def test_tavily_adapter_init_no_key_error():
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError, match="TAVILY_API_KEY not found"):
            TavilyAdapter()

def test_tavily_adapter_search_success(mock_tavily_client):
    mock_instance = mock_tavily_client.return_value
    mock_instance.search.return_value = {
        "results": [
            {"title": "Result 1", "content": "Snippet 1", "url": "http://example.com/1"},
            {"title": "Result 2", "content": "Snippet 2", "url": "http://example.com/2"}
        ]
    }
    
    adapter = TavilyAdapter(api_key="fake-key")
    response = adapter.search("test query", k=2)
    results = response["results"]
    
    assert len(results) == 2
    assert results[0]["title"] == "Result 1"
    assert results[0]["source"] == "tavily"
    assert mock_instance.search.called

def test_tavily_adapter_retry_logic(mock_tavily_client):
    mock_instance = mock_tavily_client.return_value
    # Fail twice, then succeed
    mock_instance.search.side_effect = [Exception("error"), Exception("error"), {"results": []}]
    
    adapter = TavilyAdapter(api_key="fake-key")
    with patch("time.sleep", return_value=None):  # Skip sleeping in tests
        response = adapter.search("test query")
    
    assert response["results"] == []
    assert mock_instance.search.call_count == 3
