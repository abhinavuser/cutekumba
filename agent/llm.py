from typing import Any


class LLMInterface:
    """Minimal LLM interface used by FinanceAgent.

    Implementations must provide a `generate(prompt: str) -> str` method.
    """
    def generate(self, prompt: str) -> str:
        raise NotImplementedError()


class MockLLM(LLMInterface):
    """Very small mock LLM for local testing.

    It echoes the user query and returns predictable structured JSON when it
    detects keywords like 'buy' or 'sell'. This lets the agent run without
    any external LLM dependency.
    """
    def generate(self, prompt: str) -> str:
        text = prompt.strip().lower()
        # Simple heuristic: if user asks to buy/sell return a JSON-like trade
        if text.startswith("buy ") or " buy " in text:
            # Return a realistic mock price so demos execute meaningful trades
            return '{"type":"trade","operation":"BUY","data":{"symbol":"AAPL","shares":1,"price":150.00},"natural_response":"Mock: prepared buy operation","requires_confirmation":true,"show_data":true}'
        if text.startswith("sell ") or " sell " in text:
            return '{"type":"trade","operation":"SELL","data":{"symbol":"AAPL","shares":1,"price":150.00},"natural_response":"Mock: prepared sell operation","requires_confirmation":true,"show_data":true}'

        # Default: echo back as plain text
        return "MockLLM: " + prompt


class OllamaAdapter(LLMInterface):
    """Adapter that uses LangChain's Ollama LLM if available.

    Falls back to raising an exception at construction time if the LangChain
    Ollama class isn't importable. The adapter exposes a `.generate(prompt)`
    method returning the assistant text.
    """
    def __init__(self, model: str = "llama2:7b", base_url: str = "http://localhost:11434", temperature: float = 0.3):
        try:
            from langchain_community.llms import Ollama as LC_Ollama  # type: ignore
        except Exception as e:
            raise ImportError("langchain_community Ollama is not available") from e

        # Create the underlying LangChain Ollama instance
        self._llm = LC_Ollama(model=model, temperature=temperature, base_url=base_url)

    def generate(self, prompt: str) -> str:
        # LangChain LLMs are callable. Attempt to use the standard APIs.
        try:
            result = self._llm(prompt)
            # result may be a string or an object with generations
            if isinstance(result, str):
                return result
            # Try common LangChain shape
            if hasattr(result, 'generations'):
                gens = getattr(result, 'generations')
                if gens and len(gens) > 0 and len(gens[0]) > 0:
                    text = getattr(gens[0][0], 'text', None)
                    if text:
                        return text
            # Fallback to string representation
            return str(result)
        except Exception as e:
            # Raise exceptions so callers can detect failures and fall back
            raise RuntimeError(f"Ollama/LangChain adapter error: {e}") from e


class OllamaHTTPAdapter(LLMInterface):
    """Direct HTTP adapter to Ollama's local API.

    It sends a simple prompt and returns the text response. This avoids
    requiring LangChain; it only needs the `requests` package and a local
    Ollama daemon running (default http://localhost:11434).
    """
    def __init__(self, model: str = None, base_url: str = "http://localhost:11434"):
        import requests
        self.requests = requests
        self.base_url = base_url.rstrip('/')
        # Model name: if not provided, default to llama2:latest (common installation)
        self.model = model or "llama2:latest"

    def generate(self, prompt: str) -> str:
        # Use Ollama's standard /api/generate endpoint
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }

        try:
            resp = self.requests.post(url, json=payload, timeout=30)
            resp.raise_for_status()
            
            try:
                data = resp.json()
            except Exception as e:
                raise RuntimeError(f"Failed to parse JSON response: {e}")

            # Ollama API returns response in 'response' field
            if isinstance(data, dict):
                # Check for Ollama's standard response format
                if 'response' in data and isinstance(data['response'], str):
                    return data['response']
                # Fallback to other possible fields
                if 'text' in data and isinstance(data['text'], str):
                    return data['text']
                if 'output' in data and isinstance(data['output'], str):
                    return data['output']
                # Check for OpenAI-compatible format
                choices = data.get('choices')
                if choices and isinstance(choices, list) and len(choices) > 0:
                    choice = choices[0]
                    if isinstance(choice, dict):
                        msg = choice.get('message') or {}
                        text = msg.get('content') or choice.get('text')
                        if text:
                            return text

            # If we can't parse it, return the raw text
            return resp.text

        except self.requests.exceptions.ConnectionError as e:
            raise RuntimeError(
                f"Could not connect to Ollama at {self.base_url}. "
                f"Make sure Ollama is running. Error: {e}"
            )
        except self.requests.exceptions.Timeout as e:
            raise RuntimeError(
                f"Request to Ollama timed out. The model might be loading or too slow. Error: {e}"
            )
        except self.requests.exceptions.HTTPError as e:
            if resp.status_code == 404:
                raise RuntimeError(
                    f"Model '{self.model}' not found. Available models: run 'ollama list' to see installed models. "
                    f"HTTP Error: {e}"
                )
            raise RuntimeError(f"Ollama HTTP error: {e}")
        except Exception as e:
            raise RuntimeError(f"Ollama HTTP adapter error: {e}")
