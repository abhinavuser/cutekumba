from agent.llm import OllamaHTTPAdapter, MockLLM
from agent.finance_agent import FinanceAgent
from ui.test_interface import FinanceAgentTester
import os


def main():
    # Try to use Ollama HTTP adapter if available, otherwise fallback to MockLLM
    # This uses the direct HTTP API which doesn't require LangChain
    try:
        model = os.getenv('OLLAMA_MODEL')  # Defaults to llama2:latest if not set
        llm = OllamaHTTPAdapter(model=model)
        print(f"✅ Using OllamaHTTPAdapter with model: {llm.model}")
    except Exception as e:
        print(f"⚠️ OllamaHTTPAdapter not available or failed: {e}; falling back to MockLLM")
        llm = MockLLM()

    agent = FinanceAgent(llm=llm)
    # Ensure database schema exists (delegates to SQLite fallback when Postgres not present)
    try:
        agent.db.init_db()
    except Exception as e:
        print(f"Warning: init_db failed: {e}")

    tester = FinanceAgentTester(agent=agent)
    tester.run()


if __name__ == '__main__':
    main()
