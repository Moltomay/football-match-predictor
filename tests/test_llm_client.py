import pytest
from unittest.mock import MagicMock, patch
from backend.llm.llm_client import LocalLLM

@pytest.fixture
def mock_llama():
    with patch("backend.llm.llm_client.Llama") as mock:
        yield mock

def test_llm_init_no_path_error():
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError, match="MODEL_PATH not found"):
            LocalLLM()

def test_llm_generate(mock_llama):
    with patch("os.path.exists", return_value=True):
        mock_instance = mock_llama.return_value
        mock_instance.return_value = {
            "choices": [{"text": "Hello World"}]
        }
        
        client = LocalLLM(model_path="fake/path.gguf")
        result = client.generate("Hi")
        
        assert result == "Hello World"
        assert mock_instance.called

def test_llm_generate_json_success(mock_llama):
    with patch("os.path.exists", return_value=True):
        mock_instance = mock_llama.return_value
        mock_instance.return_value = {
            "choices": [{"text": '{"key": "value"}'}]
        }
        
        client = LocalLLM(model_path="fake/path.gguf")
        result = client.generate_json("Hi")
        
        assert result == {"key": "value"}

def test_llm_generate_json_fallback(mock_llama):
    with patch("os.path.exists", return_value=True):
        mock_instance = mock_llama.return_value
        mock_instance.return_value = {
            "choices": [{"text": 'Some prefix here: {"a": 1} some suffix'}]
        }
        
        client = LocalLLM(model_path="fake/path.gguf")
        result = client.generate_json("Hi")
        
        assert result == {"a": 1}
