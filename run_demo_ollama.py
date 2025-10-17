from database.database_manager import DatabaseManager
from agent.llm import OllamaHTTPAdapter
from agent.finance_agent import FinanceAgent
import time
import uuid

# Prepare DB and agent
db = DatabaseManager()
print('Initializing DB...')
db.init_db()

# Create unique demo user
account_number = f"demo_{int(time.time())}"
email = f"{account_number}@example.com"
password = "demo_pass"
print('Creating user:', account_number)
res = db.create_user({'email': email, 'password': password, 'account_number': account_number, 'balance': 10000.0})
print('create_user ->', res)

# Instantiate agent with Ollama adapter
llm = OllamaHTTPAdapter(model='llama2:latest')
agent = FinanceAgent(llm=llm)
agent.set_current_user(account_number)

# Send a buy command
print('\nSending command: buy 1 AAPL')
resp = agent.process_request('buy 1 AAPL')
print('Agent response:\n', resp)

# If pending, confirm
if agent._pending_operation:
    print('\nConfirming pending operation...')
    out = agent.confirm_operation()
    print('Confirm result:\n', out)
else:
    print('\nNo pending operation created.')

print('\nDemo finished.')
