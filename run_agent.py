from agent.llm import OllamaAdapter, MockLLM
from agent.finance_agent import FinanceAgent
from ui.test_interface import FinanceAgentTester


def main():
    # Try to use Ollama if available, otherwise fallback to MockLLM
    try:
        llm = OllamaAdapter()
        print("Using OllamaAdapter LLM")
    except Exception as e:
        print(f"OllamaAdapter not available or failed: {e}; falling back to MockLLM")
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
