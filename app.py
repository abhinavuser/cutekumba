import sys
import types
# Some environments have a broken or partially-installed `torch` package which
# causes Streamlit's module watcher to inspect torch internals and raise a
# RuntimeError. To avoid that crash we inject a minimal dummy `torch` module
# into sys.modules before importing Streamlit. This is safe for this app
# because we don't use torch directly.
if 'torch' not in sys.modules:
    try:
        _torch_dummy = types.ModuleType('torch')
        # Provide a minimal attribute so importers that expect a __path__ don't fail
        _torch_dummy.__path__ = []
        sys.modules['torch'] = _torch_dummy
    except Exception:
        pass

import streamlit as st
from agent.finance_agent import FinanceAgent
from agent.llm import OllamaHTTPAdapter, MockLLM
from database.database_manager import DatabaseManager

# Simple Streamlit UI for FinanceAgent

@st.cache_resource
def get_agent(use_ollama: bool = True, model: str = None):
    if use_ollama:
        try:
            llm = OllamaHTTPAdapter(model=model)
            st.write(f"Using Ollama model: {llm.model}")
        except Exception as e:
            st.write(f"Ollama not available: {e}, falling back to MockLLM")
            llm = MockLLM()
    else:
        llm = MockLLM()
    agent = FinanceAgent(llm=llm)
    agent.db.init_db()
    return agent

st.title('Finance Agent')

# Attempt to use Ollama by default; if it fails we'll fallback to MockLLM
model = st.text_input('Ollama model (optional)', value='llama2:7b')
try:
    agent = get_agent(True, model)
except Exception as e:
    st.error(f"Could not initialize Ollama: {e}. Falling back to MockLLM.")
    agent = get_agent(False)

st.sidebar.header('Account')
action = st.sidebar.selectbox('Action', ['Login', 'Create'])
email = st.sidebar.text_input('Email')
password = st.sidebar.text_input('Password', type='password')
account_number = st.sidebar.text_input('Account number')

if action == 'Create' and st.sidebar.button('Create Account'):
    res = agent.db.create_user({'email': email, 'password': password, 'account_number': account_number})
    st.sidebar.write(res)

if action == 'Login' and st.sidebar.button('Login'):
    res = agent.db.validate_login(email, password)
    if res.get('status') == 'success':
        agent.set_current_user(res['data']['account_number'])
        st.sidebar.success('Logged in')
    else:
        st.sidebar.error(res.get('message'))

st.header('Portfolio')
if agent._current_user:
    st.write('Account:', agent._current_user)
    portfolio = agent.db.get_portfolio(agent._current_user)
    st.write(portfolio)
else:
    st.write('Log in to see portfolio')

st.header('Agent Chat / Commands')
query = st.text_input('Ask or command', key='query')
if st.button('Send'):
    if not agent._current_user:
        st.error('Please login first')
    else:
        resp = agent.process_request(query)
        st.write(resp)
        # If a pending operation is created and requires confirmation, show button
        if agent._pending_operation:
            if st.button('Confirm Operation'):
                out = agent.confirm_operation()
                st.write(out)

st.header('Quick actions')
col1, col2 = st.columns(2)
with col1:
    if st.button('Buy 1 AAPL'):
        if not agent._current_user:
            st.error('Login first')
        else:
            resp = agent.process_request('buy 1 AAPL')
            st.write(resp)
with col2:
    if st.button('Sell 1 AAPL'):
        if not agent._current_user:
            st.error('Login first')
        else:
            resp = agent.process_request('sell 1 AAPL')
            st.write(resp)