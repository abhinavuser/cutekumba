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
            return '{"type":"trade","operation":"BUY","data":{"symbol":"AAPL","shares":1,"price":0},"natural_response":"Mock: prepared buy operation","requires_confirmation":true,"show_data":true}'
        if text.startswith("sell ") or " sell " in text:
            return '{"type":"trade","operation":"SELL","data":{"symbol":"AAPL","shares":1,"price":0},"natural_response":"Mock: prepared sell operation","requires_confirmation":true,"show_data":true}'

        # Default: echo back as plain text
        return "MockLLM: " + prompt
