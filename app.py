import sys
import types
import os

# Workaround: inject a minimal dummy `torch` module before importing Streamlit
# to avoid Streamlit's module watcher triggering runtime errors when torch is
# partially installed or broken on the system.
if 'torch' not in sys.modules:
    try:
        _torch_dummy = types.ModuleType('torch')
        _torch_dummy.__path__ = []
        sys.modules['torch'] = _torch_dummy
    except Exception:
        pass

import streamlit as st
from agent.finance_agent import FinanceAgent
from agent.llm import OllamaHTTPAdapter, MockLLM

# set_page_config must be the first Streamlit API call in the script
st.set_page_config(page_title='Finance Agent', layout='wide')


@st.cache_resource
def get_agent(use_ollama: bool = True, model: str = None):
    """Create or return a cached FinanceAgent instance wired to an LLM.

    By default we try to use local Ollama (if available); otherwise we
    fallback to a deterministic MockLLM for offline demos.
    """
    if use_ollama:
        try:
            llm = OllamaHTTPAdapter(model=model)  # type: ignore
        except Exception:
            llm = MockLLM()
    else:
        llm = MockLLM()

    agent = FinanceAgent(llm=llm)
    # Ensure DB schema exists for the demo
    try:
        agent.db.init_db()
    except Exception:
        pass
    return agent


def ensure_session():
    if 'agent' not in st.session_state:
        model = os.getenv('OLLAMA_MODEL', 'llama2:latest')
        # Use Ollama by default if available
        try:
            st.session_state.agent = get_agent(True, model)
        except Exception:
            st.session_state.agent = get_agent(False)

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    if 'user' not in st.session_state:
        st.session_state.user = None


ensure_session()
agent: FinanceAgent = st.session_state.agent

st.title('Finance Agent — Chat')


with st.sidebar:
    st.header('Account')
    action = st.radio('Action', ['Login', 'Create'])
    email = st.text_input('Email', key='sidebar_email')
    password = st.text_input('Password', type='password', key='sidebar_password')
    account_number = st.text_input('Account number (for create)', key='sidebar_account')

    if action == 'Create' and st.button('Create Account'):
        res = agent.db.create_user({'email': email, 'password': password, 'account_number': account_number})
        st.info(res.get('message'))

    if action == 'Login' and st.button('Login'):
        res = agent.db.validate_login(email, password)
        if res.get('status') == 'success':
            st.session_state.user = res['data']['account_number']
            agent.set_current_user(st.session_state.user)
            st.success('Logged in')
        else:
            st.error(res.get('message'))

    st.markdown('---')
    st.subheader('Available commands')
    st.markdown(
        """
- buy <symbol> <shares> — e.g. buy AAPL 1
- sell <symbol> <shares>
- quote <symbol>
- portfolio
- balance
- watch <symbol>
- watchlist
- deposit <amount>
"""
    )

    st.markdown('---')
    st.caption('Note: live market data is disabled by default to avoid rate limits. Set USE_YFINANCE=1 to enable live quotes.')


col1, col2 = st.columns([3, 1])

with col1:
    st.header('Chat')

    # Render chat history
    for msg in st.session_state.chat_history:
        if msg['role'] == 'user':
            st.markdown(f"**You:** {msg['text']}")
        else:
            st.markdown(f"**Agent:** {msg['text']}")

    user_input = st.text_input('Message or command', key='chat_input')
    if st.button('Send'):
        if not st.session_state.user:
            st.error('Please login first (use the sidebar).')
        elif not user_input:
            st.warning('Type a message or command first.')
        else:
            st.session_state.chat_history.append({'role': 'user', 'text': user_input})
            resp = agent.process_request(user_input)
            st.session_state.chat_history.append({'role': 'agent', 'text': resp})

    # If a pending operation exists, show confirmation buttons
    if getattr(agent, '_pending_operation', None):
        st.warning('There is a pending operation that requires confirmation.')
        if st.button('Confirm Operation'):
            out = agent.confirm_operation()
            st.session_state.chat_history.append({'role': 'agent', 'text': out})
        if st.button('Cancel Operation'):
            agent._pending_operation = None
            st.session_state.chat_history.append({'role': 'agent', 'text': 'Operation cancelled.'})

with col2:
    st.header('Quick actions')
    if st.button('Buy 1 AAPL'):
        if not st.session_state.user:
            st.error('Login first')
        else:
            st.session_state.chat_history.append({'role': 'user', 'text': 'buy 1 AAPL'})
            resp = agent.process_request('buy 1 AAPL')
            st.session_state.chat_history.append({'role': 'agent', 'text': resp})

    if st.button('Sell 1 AAPL'):
        if not st.session_state.user:
            st.error('Login first')
        else:
            st.session_state.chat_history.append({'role': 'user', 'text': 'sell 1 AAPL'})
            resp = agent.process_request('sell 1 AAPL')
            st.session_state.chat_history.append({'role': 'agent', 'text': resp})

    st.markdown('---')
    st.header('Portfolio')
    if st.session_state.user:
        try:
            portfolio = agent.db.get_portfolio(st.session_state.user)
            balance = agent.db.get_user(st.session_state.user).get('balance')
            st.write(f"Account: {st.session_state.user}")
            st.write(f"Balance: ${balance:,.2f}")
            if portfolio:
                for p in portfolio:
                    st.write(f"- {p['stock_symbol']}: {int(p['shares'])} shares @ ${p['current_price']:.2f}")
            else:
                st.write('No holdings')
        except Exception as e:
            st.error(f"Error fetching portfolio: {e}")
    else:
        st.info('Log in to view portfolio and quick actions')

    st.markdown('---')
    st.caption('Commands: buy/sell/quote/portfolio/balance/watch/watchlist/deposit')