import os
import json
import logging
import re
from typing import Any, Dict, List
from llama_cpp import Llama
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class LocalLLM:
    """Wrapper for llama-cpp-python LLM."""

    def __init__(self, model_path: str | None = None, n_ctx: int = 4096):
        """
        Initialize the LocalLLM.

        Args:
            model_path: Path to the GGUF model file.
            n_ctx: Context window size.
        """
        self.model_path = model_path or os.getenv("MODEL_PATH")
        if not self.model_path:
            raise ValueError("MODEL_PATH not found in environment or arguments.")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at: {self.model_path}")

        try:
            self.llm = Llama(
                model_path=self.model_path,
                n_ctx=n_ctx,
                n_threads=os.cpu_count() or 4,
                verbose=False
            )
        except Exception as e:
            logger.error(f"Failed to load LLM from {self.model_path}: {e}")
            raise RuntimeError(f"LLM Load Error: {e}")

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.1) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt: The input prompt.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.

        Returns:
            The generated text.
        """
        try:
            output = self.llm(
                prompt,
                max_tokens=max_tokens,
                stop=["Q:", "\n\n"],
                echo=False,
                temperature=temperature
            )
            return output["choices"][0]["text"].strip()
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise

    def generate_json(self, prompt: str, max_tokens: int = 512) -> Dict[str, Any] | List[Any]:
        """
        Generate and parse JSON from a prompt.

        Args:
            prompt: The input prompt.
            max_tokens: Maximum tokens to generate.

        Returns:
            The parsed JSON data (dict or list).
        """
        response_text = self.generate(prompt, max_tokens=max_tokens)
        
        # Try direct parsing
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            # Fallback: extract the first balanced JSON substring
            logger.warning("Direct JSON parsing failed. Attempting regex fallback.")
            json_match = re.search(r"(\{.*\}|\[.*\])", response_text, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    pass
            
            logger.error(f"Failed to parse JSON from response: {response_text}")
            raise ValueError(f"Invalid JSON response from LLM: {response_text}")
