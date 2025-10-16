from typing import Dict, Any, List, Optional
from datetime import datetime, UTC
import json
import re
try:
    from langchain_community.llms import Ollama  # type: ignore
    from langchain.prompts import PromptTemplate  # type: ignore
    from langchain_core.output_parsers import StrOutputParser  # type: ignore
    from langchain_community.embeddings import OllamaEmbeddings  # type: ignore
    from langchain_community.vectorstores import FAISS  # type: ignore
    from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore
    from langchain.chains import RetrievalQA  # type: ignore
    _HAS_LANGCHAIN = True
except Exception:
    # LangChain / Ollama not available in the environment. We'll fallback to
    # a simple MockLLM and skip embeddings/vectorstore features.
    _HAS_LANGCHAIN = False
from database.database_manager import DatabaseManager
from agent.llm import MockLLM, LLMInterface
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import requests
from bs4 import BeautifulSoup
import time
import random

class FinanceAgent:
    def __init__(self):
        self.db = DatabaseManager()
        self.llm: LLMInterface = MockLLM()
        # Only try to wire real LLM/embeddings if langchain is available
        if _HAS_LANGCHAIN:
            try:
                self.setup_llm()
            except Exception as e:
                print(f"Could not initialize Ollama LLM, falling back to MockLLM: {e}")

            try:
                self.setup_embeddings()
                self.setup_vector_store()
            except Exception as e:
                print(f"Embeddings/Vector store unavailable: {e}")

            try:
                self.setup_prompts()
            except Exception as e:
                print(f"Could not setup prompts/chain: {e}")
        else:
            # Keep prompt/chain None when not available
            self.prompt = None
            self.chain = None
        self._pending_operation = None
        self._chat_history = []
        self._current_user = None
        self._market_data_cache = {}
        self._cache_timeout = 300  # 5 minutes

    def setup_llm(self):
        """Setup the LLM with appropriate parameters."""
        # Create Ollama-backed LLM and wrap to satisfy LLMInterface via duck typing
        self.llm = Ollama(
            model="llama2:7b",
            temperature=0.3,
            base_url="http://localhost:11434"
        )

    def setup_embeddings(self):
        """Setup embeddings for RAG."""
        # Note: embeddings may be heavy; only initialize when available
        self.embeddings = OllamaEmbeddings(
            model="llama2:13b",
            base_url="http://localhost:11434"
        )

    def setup_vector_store(self):
        """Initialize and populate vector store with financial knowledge."""
        try:
            # Load or create vector store
            try:
                self.vector_store = FAISS.load_local("financial_knowledge", self.embeddings)
            except:
                self.vector_store = self._create_knowledge_base()
        except Exception as e:
            print(f"Error setting up vector store: {e}")
            self.vector_store = None

    def _create_knowledge_base(self):
        """Create and populate the knowledge base with financial information."""
        # Financial knowledge documents
        documents = [
            # Market Analysis
            "The stock market is influenced by various factors including economic indicators, company performance, and global events.",
            "Market trends can be analyzed using technical indicators like moving averages, RSI, and MACD.",
            "Fundamental analysis involves evaluating a company's financial statements, management, and competitive advantages.",
            
            # Investment Strategies
            "Diversification is a key strategy to reduce risk by investing in different sectors and asset classes.",
            "Long-term investing typically yields better returns than short-term trading.",
            "Value investing focuses on finding undervalued stocks with strong fundamentals.",
            
            # Risk Management
            "Always invest only what you can afford to lose.",
            "Set stop-loss orders to limit potential losses.",
            "Regular portfolio rebalancing helps maintain desired risk levels.",
            
            # Market Sectors
            "Technology sector includes companies like Apple, Microsoft, and Google.",
            "Financial sector includes banks, insurance companies, and investment firms.",
            "Healthcare sector includes pharmaceutical companies and healthcare providers.",
            
            # Economic Indicators
            "GDP growth rate indicates overall economic health.",
            "Inflation rate affects purchasing power and interest rates.",
            "Unemployment rate reflects labor market conditions."
        ]

        # Create text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        # Split documents
        texts = text_splitter.split_text("\n".join(documents))

        # Create and save vector store
        vector_store = FAISS.from_texts(texts, self.embeddings)
        vector_store.save_local("financial_knowledge")
        
        return vector_store

    def setup_prompts(self):
        """Setup conversation prompts with enhanced context."""
        self.prompt = PromptTemplate(
            input_variables=["query", "current_time", "user_data", "market_data", "chat_history"],
            template="""You are an advanced AI financial assistant named FinanceGPT. You help users manage their investments, 
            execute trades, and provide financial advice. You have access to real-time market data and user portfolios.

            Current Time: {current_time}
            User Information: {user_data}
            Market Data: {market_data}
            Recent Conversation: {chat_history}

            User Query: {query}

            Your capabilities include:
            1. Natural Conversation:
               - Discuss market trends, investment strategies
               - Explain financial concepts
               - Provide personalized advice based on portfolio

            2. Account Management:
               - Create/manage user accounts
               - Show account balance and portfolio
               - Display transaction history
               - Add/remove stocks from watchlist

            3. Trading Operations:
               - Execute stock trades (buy/sell)
               - Monitor positions
               - Set price alerts
               - Analyze potential trades

            4. Market Analysis:
               - Provide real-time quotes
               - Show technical indicators
               - Discuss market news
               - Compare stocks

            When handling trades or sensitive operations:
            - ALWAYS ask for final confirmation
            - Verify account balance for purchases
            - Check existing holdings for sales
            - Show relevant market data before trades

            When responding, format trades and operations as JSON with this structure:
            {{
                "type": "conversation|account|trade|analysis",
                "operation": "CREATE|READ|UPDATE|DELETE|BUY|SELL|ANALYZE",
                "data": {{
                    "symbol": "STOCK_SYMBOL",
                    "shares": NUMBER_OF_SHARES,
                    "price": CURRENT_PRICE
                }},
                "natural_response": "Your friendly response",
                "requires_confirmation": true,
                "show_data": true
            }}

            For casual conversation, respond naturally without JSON.
            Always maintain a professional yet friendly tone.
            Base your advice on the provided market data.
            """
        )

        # In langchain-enabled environments you can wire up a chain. If not
        # available we'll fallback to using llm.generate(prompt) directly.
        if _HAS_LANGCHAIN:
            self.chain = (
                self.prompt 
                | self.llm 
                | StrOutputParser()
            )
        else:
            self.chain = None

    def get_market_data(self, symbols: List[str] = None) -> Dict:
        """Fetch comprehensive market data including indices, trends, and news."""
        try:
            current_time = time.time()
            
            # Check cache first
            if current_time - self._market_data_cache.get('timestamp', 0) < self._cache_timeout:
                return self._market_data_cache.get('data', {})

            market_data = {
                "market_status": "OPEN",
                "quotes": {},
                "indices": {
                    "^GSPC": {"name": "S&P 500"},
                    "^DJI": {"name": "Dow Jones"},
                    "^IXIC": {"name": "NASDAQ"}
                }
            }

            # Get market indices
            for symbol in market_data["indices"].keys():
                try:
                    quote = self._get_cached_quote(symbol)
                    market_data["indices"][symbol].update(quote)
                except Exception as e:
                    print(f"Error fetching index {symbol}: {e}")

            # Get specific stock quotes if requested
            if symbols:
                for symbol in symbols:
                    try:
                        quote = self._get_cached_quote(symbol)
                        market_data["quotes"][symbol] = quote
                    except Exception as e:
                        print(f"Error fetching {symbol}: {e}")

            # Cache the results
            self._market_data_cache = {
                'timestamp': current_time,
                'data': market_data
            }

            return market_data

        except Exception as e:
            print(f"Error getting market data: {e}")
            return {"market_status": "ERROR", "quotes": {}, "indices": {}}

    def _get_market_news(self) -> List[Dict]:
        """Fetch recent market news."""
        try:
            # Using a financial news API (you'll need to replace with your preferred source)
            # This is a placeholder implementation
            news = [
                {
                    "title": "Market Update",
                    "summary": "Markets show mixed performance today",
                    "source": "Financial News",
                    "timestamp": datetime.now(UTC).isoformat()
                }
            ]
            return news
        except Exception as e:
            print(f"Error fetching news: {e}")
            return []

    def analyze_market_trends(self) -> Dict:
        """Analyze current market trends and provide insights."""
        try:
            market_data = self.get_market_data()
            
            # Calculate market sentiment
            indices = market_data.get("indices", {})
            avg_change = sum(float(data.get('change', 0)) for data in indices.values()) / len(indices)
            
            # Analyze sector performance
            sectors = market_data.get("sector_performance", {})
            sector_trends = {
                "leading": [],
                "lagging": []
            }
            
            for sector, data in sectors.items():
                change = float(data.get('change', 0))
                if change > 1.0:
                    sector_trends["leading"].append(sector)
                elif change < -1.0:
                    sector_trends["lagging"].append(sector)

            return {
                "market_sentiment": "Bullish" if avg_change > 0.5 else "Bearish" if avg_change < -0.5 else "Neutral",
                "sector_trends": sector_trends,
                "market_summary": self._generate_market_summary(market_data)
            }
        except Exception as e:
            print(f"Error analyzing market trends: {e}")
            return {}

    def _generate_market_summary(self, market_data: Dict) -> str:
        """Generate a comprehensive market summary."""
        try:
            summary = []
            
            # Add market status
            market_time = datetime.now(UTC)
            is_market_open = (
                market_time.weekday() < 5 and
                13 <= market_time.hour <= 20
            )
            summary.append(f"Market Status: {'🟢 OPEN' if is_market_open else '🔴 CLOSED'}")
            
            # Add major indices
            indices = market_data.get("indices", {})
            for symbol, data in indices.items():
                change = float(data.get('change', 0))
                emoji = "📈" if change > 0 else "📉" if change < 0 else "➖"
                summary.append(
                    f"{emoji} {data['name']}: ${float(data.get('price', 0)):,.2f} "
                    f"({change:+.2f}%)"
                )
            
            # Add sector performance
            sectors = market_data.get("sector_performance", {})
            if sectors:
                summary.append("\nSector Performance:")
                for sector, data in sectors.items():
                    change = float(data.get('change', 0))
                    emoji = "📈" if change > 0 else "📉" if change < 0 else "➖"
                    summary.append(f"{emoji} {sector}: {change:+.2f}%")
            
            return "\n".join(summary)
        except Exception as e:
            print(f"Error generating market summary: {e}")
            return "Error generating market summary"

    def get_investment_recommendations(self, amount: float) -> List[Dict]:
        """Get personalized investment recommendations based on amount and market conditions."""
        try:
            market_data = self.get_market_data()
            trends = self.analyze_market_trends()
            
            recommendations = []
            
            # Get top performing sectors
            sectors = market_data.get("sector_performance", {})
            top_sectors = sorted(
                sectors.items(),
                key=lambda x: float(x[1].get('change', 0)),
                reverse=True
            )[:3]
            
            # Get stocks from top sectors
            for sector, _ in top_sectors:
                sector_stocks = self._get_sector_stocks(sector)
                for stock in sector_stocks[:2]:  # Top 2 stocks from each sector
                    try:
                        quote = self._get_cached_quote(stock)
                        if quote and 'price' in quote:
                            price = float(quote['price'])
                            shares = int(amount * 0.2 / price)  # 20% of amount per stock
                            if shares > 0:
                                recommendations.append({
                                    "symbol": stock,
                                    "name": self._get_stock_name(stock),
                                    "price": price,
                                    "shares": shares,
                                    "total": price * shares,
                                    "sector": sector,
                                    "reason": f"Strong performer in {sector} sector"
                                })
                    except Exception as e:
                        print(f"Error getting quote for {stock}: {e}")
            
            return recommendations
        except Exception as e:
            print(f"Error getting recommendations: {e}")
            return []

    def _get_sector_stocks(self, sector: str) -> List[str]:
        """Get list of stocks in a sector."""
        # This is a simplified implementation
        sector_stocks = {
            "Technology": ["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
            "Financial": ["JPM", "BAC", "GS", "V", "MA"],
            "Healthcare": ["JNJ", "PFE", "UNH", "ABBV"],
            "Energy": ["XOM", "CVX"],
            "Industrial": ["CAT", "BA", "GE"],
            "Consumer": ["WMT", "COST", "PG", "KO", "PEP"],
            "Entertainment": ["DIS", "NFLX", "CMCSA"]
        }
        return sector_stocks.get(sector, [])

    def _get_stock_name(self, symbol: str) -> str:
        """Get company name for a stock symbol."""
        try:
            stock = yf.Ticker(symbol)
            return stock.info.get('longName', symbol)
        except:
            return symbol

    def process_request(self, query: str) -> str:
        """Process user requests with enhanced context."""
        try:
            # Get current context
            current_time = datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')
            
            # Get user data if available
            user_data = {}
            if self._current_user:
                try:
                    portfolio = self.db.get_portfolio(self._current_user)
                    watchlist = self.db.get_watchlist(self._current_user)
                    user = self.db.get_user(self._current_user)
                    user_data = {
                        "account_number": self._current_user,
                        "balance": float(user['balance']),
                        "portfolio": [
                            {
                                "symbol": p['stock_symbol'],
                                "shares": int(p['shares']),
                                "avg_price": float(p['average_price']),
                                "current_price": float(p.get('current_price', 0))
                            } for p in portfolio
                        ],
                        "watchlist": [w['stock_symbol'] for w in watchlist]
                    }
                except Exception as e:
                    print(f"Error getting user data: {e}")

            # Get market data
            market_data = self.get_market_data()
            
            # Process through LLM with enhanced context
            response = self.chain.invoke({
                "query": query,
                "current_time": current_time,
                "user_data": json.dumps(user_data, default=str),
                "market_data": json.dumps(market_data, default=str),
                "chat_history": "\n".join(self._chat_history[-5:])
            })
            
            # Save chat history
            if self._current_user:
                try:
                    self.db.save_chat_message(self._current_user, "USER", query)
                    self.db.save_chat_message(self._current_user, "ASSISTANT", response)
                    self._chat_history.append(f"User: {query}")
                    self._chat_history.append(f"Assistant: {response}")
                except Exception as e:
                    print(f"Error saving chat history: {e}")
            
            return response
            
        except Exception as e:
            print(f"Error processing request: {e}")
            return "I'm having trouble processing that request. Please try again or use a specific command."

    def _get_cached_quote(self, symbol: str) -> Dict:
        """Get quote from cache or fetch new data."""
        current_time = time.time()
        
        # Check cache first
        if symbol in self._market_data_cache.get('quotes', {}):
            quote = self._market_data_cache['quotes'][symbol]
            if current_time - quote.get('timestamp', 0) < self._cache_timeout:
                return quote

        # Add delay between requests
        time.sleep(0.5)
        
        # Fetch new quote
        quote = self.db.get_real_time_quote(symbol)
        quote['timestamp'] = current_time
        return quote

    def analyze_sentiment(self, text: str) -> Dict:
        """Analyze if text indicates buying, selling, or general inquiry."""
        text = text.lower()
        
        # Common patterns for trading intentions
        buy_patterns = ['buy', 'purchase', 'invest in', 'get some', 'acquire']
        sell_patterns = ['sell', 'dump', 'get rid of', 'dispose', 'exit']
        
        # Common patterns for casual conversation
        greeting_patterns = ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening']
        question_patterns = ['how', 'what', 'why', 'when', 'where', 'can you', 'could you']
        
        # Check for casual conversation first
        if any(pattern in text for pattern in greeting_patterns):
            return {"action": "CHAT", "type": "greeting"}
        
        if any(pattern in text for pattern in question_patterns):
            return {"action": "CHAT", "type": "question"}
        
        # Check for trading patterns
        for pattern in buy_patterns:
            if pattern in text:
                symbols = re.findall(r'[A-Z]{1,5}', text.upper())
                if symbols:
                    return {"action": "BUY", "symbol": symbols[0]}
        
        for pattern in sell_patterns:
            if pattern in text:
                symbols = re.findall(r'[A-Z]{1,5}', text.upper())
                if symbols:
                    return {"action": "SELL", "symbol": symbols[0]}
        
        return {"action": "ANALYZE"}

    def set_current_user(self, account_number: str):
        """Set the current user context."""
        self._current_user = account_number

    def _handle_trade_command(self, query: str) -> str:
        """Handle buy/sell trade commands."""
        try:
            if not self._current_user:
                return "❌ Please log in to execute trades."

            words = query.lower().split()
            action = 'BUY' if 'buy' in words else 'SELL'
            symbol_idx = words.index('buy' if 'buy' in words else 'sell') + 1
            
            # Extract symbol and shares
            shares = None
            symbol = None
            
            # Define supported symbols
            SUPPORTED_SYMBOLS = {
                # Technology
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'AMD', 'INTC', 'CSCO',
                # Financial
                'JPM', 'BAC', 'GS', 'V', 'MA',
                # Consumer
                'WMT', 'COST', 'PG', 'KO', 'PEP', 'MCD',
                # Entertainment
                'DIS', 'NFLX',
                # Other sectors
                'TSLA', 'F', 'GM', 'GE', 'XOM', 'CVX', 'T', 'VZ'
            }
            
            # Parse query for shares and symbol
            for word in words[symbol_idx:]:
                if word.isdigit():
                    shares = int(word)
                elif word.upper() in SUPPORTED_SYMBOLS:
                    symbol = word.upper()
            
            if not symbol:
                return (
                    "❌ Invalid or missing stock symbol.\n"
                    "Use 'symbols' command to see supported stocks."
                )
                
            if not shares:
                return (
                    "❌ Please specify the number of shares.\n"
                    f"Example: {action.lower()} {symbol} 10"
                )
            
            if shares <= 0:
                return "❌ Number of shares must be positive."
            
            # Get current price
            quote = self._get_cached_quote(symbol)
            if not quote or 'price' not in quote:
                return f"❌ Error: Unable to get quote for {symbol}"
            
            total_cost = float(quote['price']) * shares
            
            # Verify sufficient balance for buy orders
            if action == 'BUY':
                try:
                    user = self.db.get_user(self._current_user)
                    if float(user['balance']) < total_cost:
                        return (
                            f"❌ Insufficient funds for this trade.\n"
                            f"Required: ${total_cost:,.2f}\n"
                            f"Available: ${float(user['balance']):,.2f}"
                        )
                except Exception as e:
                    return f"❌ Error verifying account balance: {str(e)}"
            
            # Verify sufficient shares for sell orders
            if action == 'SELL':
                try:
                    portfolio = self.db.get_portfolio(self._current_user)
                    position = next((pos for pos in portfolio if pos['stock_symbol'] == symbol), None)
                    if not position or int(position['shares']) < shares:
                        available = position['shares'] if position else 0
                        return (
                            f"❌ Insufficient shares for this trade.\n"
                            f"Required: {shares} shares\n"
                            f"Available: {available} shares"
                        )
                except Exception as e:
                    return f"❌ Error verifying share holdings: {str(e)}"
            
            # Prepare trade data
            trade_data = {
                "type": "trade",
                "operation": action,
                "data": {
                    "symbol": symbol,
                    "shares": shares,
                    "price": float(quote['price'])
                },
                "natural_response": (
                    f"Would you like to {action.lower()} {shares} shares of {symbol} "
                    f"at ${float(quote['price']):.2f} per share?"
                ),
                "requires_confirmation": True,
                "show_data": True
            }
            
            self._pending_operation = trade_data
            
            # Build detailed response
            response = [
                f"💹 {symbol} Trade Confirmation",
                f"Action: {action}",
                f"Shares: {shares:,}",
                f"Price: ${float(quote['price']):.2f}",
                f"Total {'cost' if action == 'BUY' else 'proceeds'}: ${total_cost:,.2f}",
                f"Change Today: {float(quote.get('change', 0)):.2f}%",
                f"Volume: {int(quote.get('volume', 0)):,}",
                "",
                "Please confirm by saying 'yes' or 'confirm'"
            ]
            
            return "\n".join(response)
            
        except ValueError as ve:
            return f"❌ Invalid trade command: {str(ve)}\nExample: buy AAPL 10"
        except Exception as e:
            print(f"Trade error: {str(e)}")  # Log error for debugging
            return "❌ Error processing trade. Please try again with format: buy/sell SYMBOL SHARES"

    def _get_stock_quote(self, symbol: str) -> str:
        """Get and format stock quote."""
        try:
            quote = self._get_cached_quote(symbol)
            return (
                f"\n📈 {symbol} Quote:\n"
                f"Price: ${float(quote['price']):.2f}\n"
                f"Change: {float(quote['change']):.2f}%\n"
                f"Volume: {int(quote['volume']):,}"
            )
        except Exception as e:
            return f"❌ Error getting quote for {symbol}: {str(e)}"

    def _get_balance(self) -> str:
        """Get user's current balance."""
        if not self._current_user:
            return "Please log in to check your balance."
        try:
            user = self.db.get_user(self._current_user)
            return f"💰 Current Balance: ${float(user['balance']):,.2f}"
        except Exception as e:
            return f"Error getting balance: {str(e)}"

    def _get_watchlist(self) -> str:
        """Get user's watchlist with current prices."""
        if not self._current_user:
            return "Please log in to view your watchlist."
        try:
            watchlist = self.db.get_watchlist(self._current_user)
            if not watchlist:
                return (
                    "📋 Your Watchlist is empty\n"
                    "Add stocks using 'add <SYMBOL> to watchlist' or 'watch <SYMBOL>'"
                )
            
            lines = ["📋 Your Watchlist:"]
            total_change = 0
            
            for item in watchlist:
                try:
                    quote = self._get_cached_quote(item['stock_symbol'])
                    change = float(quote.get('change', 0))
                    price = float(quote.get('price', 0))
                    emoji = "📈" if change > 0 else "📉" if change < 0 else "➖"
                    
                    lines.append(
                        f"{emoji} {item['stock_symbol']}: ${price:.2f} "
                        f"({change:+.2f}%)"
                    )
                    total_change += change
                except Exception as e:
                    lines.append(f"❌ {item['stock_symbol']}: Error getting quote")
            
            # Add summary if there are items
            if len(watchlist) > 0:
                avg_change = total_change / len(watchlist)
                lines.append(f"\nAverage Change: {avg_change:+.2f}%")
            
            return "\n".join(lines)
        except Exception as e:
            return f"Error getting watchlist: {str(e)}"

    def _get_help(self) -> str:
        """Get help message with available commands."""
        return """
📌 Available Commands:
1. login <email> <password> - Login to account
2. create <email> <password> <account_number> - Create new account
3. portfolio - View your portfolio
4. quote <symbol> - Get stock quote
5. buy <symbol> <shares> - Buy stocks
6. sell <symbol> <shares> - Sell stocks
7. watch <symbol> - Add to watchlist
8. watchlist - View watchlist
9. balance - Check balance
10. deposit <amount> - Deposit funds
11. chat <message> - Chat with agent
12. clear - Clear screen
13. exit - Exit application

You can also ask questions naturally!
"""

    def _execute_operation(self, operation: Dict) -> str:
        """Execute parsed operations with proper error handling."""
        try:
            if not self._current_user:
                return "❌ Please log in to perform this operation."

            if operation["type"] == "trade":
                result = self.db.execute_trade(
                    self._current_user,
                    operation["operation"],
                    operation["data"]["symbol"],
                    operation["data"]["shares"],
                    operation["data"]["price"]
                )
                
                if result["status"] == "success":
                    # Update portfolio summary after trade
                    portfolio = self.db.get_portfolio(self._current_user)
                    return f"✅ {result['message']}\n\n{self._format_portfolio_summary(portfolio)}"
                return f"❌ {result['message']}"
            
            elif operation["type"] == "account":
                if operation["operation"] == "READ":
                    if "portfolio" in operation["data"]:
                        portfolio = self.db.get_portfolio(self._current_user)
                        return self._format_portfolio_summary(portfolio)
                    elif "watchlist" in operation["data"]:
                        watchlist = self.db.get_watchlist(self._current_user)
                        return "Watchlist:\n" + "\n".join(
                            f"- {item['stock_symbol']}: ${item['price']:.2f}" 
                            for item in watchlist
                        )
            
            return operation["natural_response"]
            
        except Exception as e:
            return f"❌ Error executing operation: {str(e)}"

    def confirm_operation(self) -> str:
        """Execute a pending operation after user confirmation."""
        if self._pending_operation:
            operation = self._pending_operation
            self._pending_operation = None
            return self._execute_operation(operation)
        return "No pending operation to confirm."

    def get_portfolio_summary(self) -> str:
        """Get formatted portfolio summary for current user."""
        if not self._current_user:
            return "Please log in to view portfolio."
        
        try:
            portfolio = self.db.get_portfolio(self._current_user)
            if not portfolio:
                return "Your portfolio is empty."
            
            total_value = sum(float(pos['shares']) * float(pos['current_price']) for pos in portfolio)
            total_cost = sum(float(pos['shares']) * float(pos['average_price']) for pos in portfolio)
            total_pl = total_value - total_cost
            
            summary = [
                "📊 Portfolio Summary:",
                f"Total Value: ${total_value:,.2f}",
                f"Total P/L: ${total_pl:,.2f} ({(total_pl/total_cost)*100:.2f}% overall)\n",
                "Current Positions:"
            ]
            
            for pos in portfolio:
                current_value = float(pos['shares']) * float(pos['current_price'])
                cost_basis = float(pos['shares']) * float(pos['average_price'])
                position_pl = current_value - cost_basis
                pl_percent = (position_pl / cost_basis) * 100
                
                summary.append(
                    f"- {pos['stock_symbol']}: {int(pos['shares'])} shares @ ${float(pos['average_price']):.2f} "
                    f"(Current: ${float(pos['current_price']):.2f}, P/L: ${position_pl:.2f} / {pl_percent:.2f}%)"
                )
            
            return "\n".join(summary)
        except Exception as e:
            print(f"Error formatting portfolio: {str(e)}")
            return "Error displaying portfolio."

    def _format_portfolio_summary(self, portfolio: List[Dict]) -> str:
        """Format portfolio data for display."""
        if not portfolio:
            return "Your portfolio is empty."
        
        try:
            # Convert Decimal to float for calculations
            total_value = sum(float(pos['shares']) * float(pos['current_price']) for pos in portfolio)
            total_cost = sum(float(pos['shares']) * float(pos['average_price']) for pos in portfolio)
            total_pl = total_value - total_cost
            
            summary = [
                "📊 Portfolio Summary:",
                f"Total Value: ${total_value:,.2f}",
                f"Total P/L: ${total_pl:,.2f} ({(total_pl/total_cost)*100:.2f}% overall)\n",
                "Current Positions:"
            ]
            
            # Group positions by symbol to avoid duplicates
            positions = {}
            for pos in portfolio:
                symbol = pos['stock_symbol']
                if symbol not in positions:
                    positions[symbol] = {
                        'shares': float(pos['shares']),
                        'average_price': float(pos['average_price']),
                        'current_price': float(pos.get('current_price', pos['average_price']))
                    }
                else:
                    positions[symbol]['shares'] += float(pos['shares'])
            
            for symbol, pos in positions.items():
                current_value = pos['shares'] * pos['current_price']
                cost_basis = pos['shares'] * pos['average_price']
                position_pl = current_value - cost_basis
                pl_percent = (position_pl / cost_basis * 100) if cost_basis != 0 else 0
                
                summary.append(
                    f"- {symbol}: {int(pos['shares'])} shares @ ${pos['average_price']:.2f} "
                    f"(Current: ${pos['current_price']:.2f}, P/L: ${position_pl:.2f} / {pl_percent:.2f}%)"
                )
            
            return "\n".join(summary)
        except Exception as e:
            print(f"Error formatting portfolio: {e}")
            return "Error displaying portfolio."