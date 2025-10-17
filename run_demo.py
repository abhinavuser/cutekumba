from database.database_manager import DatabaseManager
from agent.llm import MockLLM
from agent.finance_agent import FinanceAgent
import time

# Demo: init DB, create unique user, run a mock buy flow and print DB tables

def run_demo():
    db = DatabaseManager()
    db.init_db()
    print('DB initialized')

    ts = str(int(time.time()))
    acct = f'demo{ts}'
    email = f'demo{ts}@local'
    # Create user with explicit starting balance to allow buys
    res = db.create_user({'email': email, 'password': 'demo123', 'account_number': acct, 'balance': 10000.0})
    print('create_user ->', res)

    # Use MockLLM for deterministic demo
    llm = MockLLM()
    print('Using MockLLM for demo')

    agent = FinanceAgent(llm=llm)
    agent.set_current_user(acct)
    print('Sending buy command')
    resp = agent.process_request('buy 1 AAPL')
    print('agent response ->', resp)
    if agent._pending_operation:
        print('confirming pending operation')
        print(agent.confirm_operation())

    print('\nUsers:')
    print(db.execute_query('SELECT id, account_number, email, balance FROM users'))
    print('\nPortfolio:')
    print(db.execute_query('SELECT * FROM portfolio'))
    print('\nTransactions:')
    print(db.execute_query('SELECT * FROM transactions'))

if __name__ == '__main__':
    run_demo()
from database.database_manager import DatabaseManager
from agent.llm import OllamaHTTPAdapter, MockLLM
from agent.finance_agent import FinanceAgent

# Demo: init DB, create user, run a mock buy flow and print DB tables

def run_demo():
    db = DatabaseManager()
    db.init_db()
    print('DB initialized')

    res = db.create_user({'email': 'demo@local', 'password': 'demo123', 'account_number': 'demo123'})
    print('create_user ->', res)

    # Use MockLLM for deterministic demo unless Ollama is available
    # Use MockLLM for deterministic demo
    llm = MockLLM()
    print('Using MockLLM for demo')

    agent = FinanceAgent(llm=llm)
    agent.set_current_user('demo123')

    print('Sending buy command')
    resp = agent.process_request('buy 1 AAPL')
    print('agent response ->', resp)

    if agent._pending_operation:
        print('confirming pending operation')
        print(agent.confirm_operation())

    print('\nUsers:')
    print(db.execute_query('SELECT id, account_number, email, balance FROM users'))
    print('\nPortfolio:')
    print(db.execute_query('SELECT * FROM portfolio'))
    print('\nTransactions:')
    print(db.execute_query('SELECT * FROM transactions'))

if __name__ == '__main__':
    run_demo()
