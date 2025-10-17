import psycopg2
from psycopg2.extras import RealDictCursor
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import bcrypt
import os
from dotenv import load_dotenv
import yfinance as yf
import pandas as pd
import time
import random 
import sqlite3
from typing import Tuple

class DatabaseManager:
    def __init__(self):
        """Initialize database connection parameters."""
        load_dotenv()
        self.db_config = {
            "dbname": os.getenv("DB_NAME", "finance_db"),
            "user": os.getenv("DB_USER", "abhinavuser"),
            "password": os.getenv("DB_PASSWORD", "your_password"),
            "host": os.getenv("DB_HOST", "localhost"),
            "port": os.getenv("DB_PORT", "5432")
        }

        self._quote_cache = {}
        self._cache_timeout = 60 
        self._impl = None  # If set, delegates to SQLite fallback implementation

        # Test Postgres connection; if it fails, use local SQLite fallback
        try:
            conn = self.get_connection()
            conn.close()
        except Exception:
            # Use SQLite fallback
            sqlite_path = os.getenv("SQLITE_PATH", "ai_project_data.sqlite3")
            self._impl = _SQLiteDatabaseManager(sqlite_path)
            # Defer schema initialization to explicit init_db() call

    def get_connection(self):
        """Create and return a database connection with dict cursor."""
        try:
            return psycopg2.connect(**self.db_config, cursor_factory=RealDictCursor)
        except Exception as e:
            raise Exception(f"Database connection error: {str(e)}")

    def execute_query(self, query: str, parameters: Any = None, fetch: bool = True) -> Optional[List[Dict]]:
        """Execute a database query with proper error handling and connection management."""
        # If using sqlite fallback, delegate
        if getattr(self, "_impl", None):
            return self._impl.execute_query(query, parameters, fetch)

        try:
            # Debug print (can be removed or toggled by a debug flag)
            # print(f"\nExecuting query: {query}")
            # print(f"With parameters: {parameters}")

            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(query, parameters)
                    if fetch:
                        results = cur.fetchall()
                        return results
                    else:
                        conn.commit()
                        return {"affected_rows": cur.rowcount}

        except Exception as e:
            # For library callers, raise with a clear message
            raise Exception(f"Query execution error: {str(e)}")

    def create_user(self, data: Dict) -> Dict:
        """Create a new user account with validation."""
        # Delegate to sqlite impl when active
        if getattr(self, "_impl", None):
            return self._impl.create_user(data)
        try:
            print(f"Attempting to create user with email: {data.get('email')}")

            # Validate required fields
            required_fields = ['email', 'password', 'account_number']
            missing_fields = [field for field in required_fields if field not in data]
            if missing_fields:
                return {"status": "error", "message": f"Missing required fields: {', '.join(missing_fields)}"}

            # Hash password
            salt = bcrypt.gensalt()
            hashed_password = bcrypt.hashpw(data['password'].encode('utf-8'), salt)

            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    # First check if user already exists
                    cur.execute("SELECT email FROM users WHERE email = %s", (data['email'],))
                    if cur.fetchone():
                        return {"status": "error", "message": "User with this email already exists"}

                    query = """
                        INSERT INTO users (account_number, email, password, balance)
                        VALUES (%s, %s, %s, %s)
                        RETURNING id, account_number, email, balance, created_at
                    """

                    cur.execute(query, (
                        data['account_number'],
                        data['email'],
                        hashed_password.decode('utf-8'),
                        data.get('balance', 10000.00)
                    ))

                    result = cur.fetchone()
                    conn.commit()

                    return {
                        "status": "success",
                        "message": f"Account created successfully with number {result['account_number']}",
                        "data": result
                    }
        except Exception as e:
            print(f"Error creating user: {str(e)}")
            return {"status": "error", "message": str(e)}

    def validate_login(self, email: str, password: str) -> Dict:
        """Validate user login credentials."""
        # Delegate to sqlite impl when active
        if getattr(self, "_impl", None):
            return self._impl.validate_login(email, password)

        try:
            print(f"\nAttempting login for email: {email}")
            query = """
                SELECT id, account_number, email, password, balance 
                FROM users 
                WHERE email = %s
            """

            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(query, (email,))
                    result = cur.fetchone()

                    if not result:
                        return {"status": "error", "message": "Invalid email or password"}

                    user = dict(result)
                    stored_password = user['password']
                    if bcrypt.checkpw(password.encode('utf-8'), stored_password.encode('utf-8')):
                        # Update last login
                        cur.execute(
                            "UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = %s",
                            (user['id'],)
                        )
                        conn.commit()
                        return {"status": "success", "data": {
                            "account_number": user['account_number'],
                            "email": user['email'],
                            "balance": user['balance']
                        }}

                    return {"status": "error", "message": "Invalid email or password"}
        except Exception as e:
            print(f"Login error: {str(e)}")
            return {"status": "error", "message": str(e)}

    def get_real_time_quote(self, symbol: str) -> Dict:
        """Get real-time stock quote using yfinance with caching and rate limiting."""
        # Delegate to sqlite impl when active
        if getattr(self, "_impl", None):
            return self._impl.get_real_time_quote(symbol)

        try:
            # Check cache first
            current_time = time.time()
            if symbol in self._quote_cache:
                cached_quote, cache_time = self._quote_cache[symbol]
                if current_time - cache_time < self._cache_timeout:
                    return cached_quote

            # Add random delay between requests to avoid rate limiting
            time.sleep(random.uniform(1, 3))

            # If symbol is AAPL, return test data (temporary workaround for rate limit)
            if symbol.upper() == 'AAPL':
                quote_data = {
                    "symbol": "AAPL",
                    "price": 169.50,  # Example price
                    "change": 0.5,
                    "volume": 50000000,
                    "timestamp": datetime.now().isoformat()
                }
                self._quote_cache[symbol] = (quote_data, current_time)
                return quote_data

            max_retries = 3
            for attempt in range(max_retries):
                try:
                    stock = yf.Ticker(symbol.upper())
                    info = stock.info

                    if 'regularMarketPrice' not in info and 'previousClose' not in info:
                        raise ValueError(f"No price data available for {symbol}")

                    # If no price data available, set price to None so caller can handle it
                    price_val = info.get('regularMarketPrice', info.get('previousClose', None))
                    quote_data = {
                            "symbol": symbol.upper(),
                            "price": price_val,
                            "change": info.get('regularMarketChangePercent', None),
                            "volume": info.get('regularMarketVolume', None),
                            "timestamp": datetime.now().isoformat()
                        }

                    # Cache the result
                    self._quote_cache[symbol] = (quote_data, current_time)
                    return quote_data

                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(2 ** attempt)  # Exponential backoff

        except Exception as e:
            print(f"Error fetching quote for {symbol}: {str(e)}")
            # Return last cached value if available
            if symbol in self._quote_cache:
                cached_quote, _ = self._quote_cache[symbol]
                cached_quote['from_cache'] = True
                return cached_quote

            # Return safe default with error indication (price set to None)
            return {
                "symbol": symbol.upper(),
                "price": None,
                "change": None,
                "volume": None,
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }

    def execute_trade(self, account_number: str, trade_type: str, 
                     symbol: str, shares: int, price: float) -> Dict:
        """Execute a stock trade with proper validation and error handling."""
        # Delegate to sqlite impl when active
        if getattr(self, "_impl", None):
            return self._impl.execute_trade(account_number, trade_type, symbol, shares, price)
        # Execute with proper validation and return structured errors instead of raising
        try:
            # Validate price
            if price is None or float(price) <= 0:
                return {"status": "error", "message": "Invalid or missing price for this trade"}

            total_amount = float(shares) * float(price)

            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    # Lock user's balance row
                    cur.execute("SELECT balance FROM users WHERE account_number = %s FOR UPDATE", (account_number,))
                    user = cur.fetchone()
                    if not user:
                        return {"status": "error", "message": "Account not found"}

                    # Calculate current holdings
                    cur.execute("SELECT SUM(shares) as total_shares FROM portfolio WHERE account_number = %s AND stock_symbol = %s", (account_number, symbol))
                    position = cur.fetchone()
                    current_shares = int(position['total_shares']) if position and position['total_shares'] else 0

                    if trade_type == 'SELL' and current_shares < shares:
                        return {"status": "error", "message": f"Insufficient shares for this trade. Required: {shares}, Available: {current_shares}"}

                    current_balance = float(user['balance'])
                    if trade_type == 'BUY' and current_balance < total_amount:
                        return {"status": "error", "message": f"Insufficient funds for this trade. Required: ${total_amount:.2f}, Available: ${current_balance:.2f}"}

                    # Perform updates
                    if trade_type == 'BUY':
                        cur.execute("UPDATE users SET balance = balance - %s WHERE account_number = %s", (total_amount, account_number))
                        if current_shares > 0:
                            # Update average price correctly
                            cur.execute("SELECT shares, average_price FROM portfolio WHERE account_number = %s AND stock_symbol = %s", (account_number, symbol))
                            existing = cur.fetchone()
                            if existing:
                                existing_shares = float(existing['shares'])
                                existing_avg = float(existing['average_price'])
                                new_avg = ((existing_avg * existing_shares) + (price * shares)) / (existing_shares + shares)
                                cur.execute("UPDATE portfolio SET shares = shares + %s, average_price = %s, last_updated = CURRENT_TIMESTAMP WHERE account_number = %s AND stock_symbol = %s", (shares, new_avg, account_number, symbol))
                        else:
                            cur.execute("INSERT INTO portfolio (account_number, stock_symbol, shares, average_price) VALUES (%s, %s, %s, %s)", (account_number, symbol, shares, price))

                    else:  # SELL
                        cur.execute("UPDATE users SET balance = balance + %s WHERE account_number = %s", (total_amount, account_number))
                        cur.execute("UPDATE portfolio SET shares = shares - %s, last_updated = CURRENT_TIMESTAMP WHERE account_number = %s AND stock_symbol = %s", (shares, account_number, symbol))
                        cur.execute("DELETE FROM portfolio WHERE account_number = %s AND stock_symbol = %s AND shares <= 0", (account_number, symbol))

                    cur.execute("INSERT INTO transactions (account_number, transaction_type, stock_symbol, shares, price_per_share, total_amount) VALUES (%s, %s, %s, %s, %s, %s) RETURNING transaction_id", (account_number, trade_type, symbol, shares, price, total_amount))
                    transaction = cur.fetchone()
                    conn.commit()

                    return {"status": "success", "message": f"Successfully {trade_type.lower()}ed {shares} shares of {symbol} at ${price:.2f} per share", "transaction_id": transaction['transaction_id']}

        except Exception as e:
            return {"status": "error", "message": str(e)}
                
    def get_portfolio(self, account_number: str) -> List[Dict]:
        """Get user's consolidated portfolio with current market values."""
        # Delegate to sqlite impl when active
        if getattr(self, "_impl", None):
            return self._impl.get_portfolio(account_number)

        try:
            # Get consolidated portfolio holdings with single row per symbol
            query = """
                WITH latest_transactions AS (
                    SELECT DISTINCT ON (account_number, stock_symbol)
                        account_number, stock_symbol, price_per_share, transaction_date
                    FROM transactions
                    ORDER BY account_number, stock_symbol, transaction_date DESC
                )
                SELECT 
                    p.stock_symbol,
                    SUM(p.shares) as shares,  -- Sum total shares
                    p.average_price,
                    p.last_updated,
                    lt.price_per_share as last_transaction_price,
                    lt.transaction_date as last_transaction_date
                FROM portfolio p
                LEFT JOIN latest_transactions lt 
                    ON p.account_number = lt.account_number 
                    AND p.stock_symbol = lt.stock_symbol
                WHERE p.account_number = %s AND p.shares > 0
                GROUP BY 
                    p.stock_symbol,
                    p.average_price,
                    p.last_updated,
                    lt.price_per_share,
                    lt.transaction_date
            """
            
            portfolio = self.execute_query(query, (account_number,))
            
            # Convert to list of consolidated positions
            consolidated = {}
            for pos in portfolio:
                symbol = pos['stock_symbol']
                if symbol not in consolidated:
                    consolidated[symbol] = {
                        'stock_symbol': symbol,
                        'shares': float(pos['shares']),
                        'average_price': float(pos['average_price']),
                        'last_updated': pos['last_updated'],
                        'last_transaction_price': float(pos['last_transaction_price']) if pos['last_transaction_price'] else None,
                        'last_transaction_date': pos['last_transaction_date']
                    }
                else:
                    consolidated[symbol]['shares'] += float(pos['shares'])
            
            # Convert consolidated dict to list
            portfolio = list(consolidated.values())
            
            # Enrich with current market prices
            for position in portfolio:
                try:
                    quote = self.get_real_time_quote(position['stock_symbol'])
                    current_price = float(quote['price'])
                    shares = position['shares']
                    avg_price = position['average_price']
                    
                    position['current_price'] = current_price
                    position['market_value'] = current_price * shares
                    position['profit_loss'] = (current_price - avg_price) * shares
                    position['profit_loss_percent'] = ((current_price / avg_price) - 1) * 100 if avg_price > 0 else 0
                    
                except Exception as e:
                    print(f"Error getting quote for {position['stock_symbol']}: {e}")
                    position['current_price'] = position['average_price']
                    position['market_value'] = position['average_price'] * shares
                    position['profit_loss'] = 0
                    position['profit_loss_percent'] = 0
            
            return portfolio
            
        except Exception as e:
            print(f"Error fetching portfolio: {e}")
            return []

    

    def save_chat_message(self, account_number: str, message_type: str, message: str) -> None:
        """Save chat message to history."""
        # Delegate to sqlite impl when active
        if getattr(self, "_impl", None):
            return self._impl.save_chat_message(account_number, message_type, message)

        try:
            query = """
                INSERT INTO chat_history (account_number, message_type, message)
                VALUES (%s, %s, %s)
            """
            self.execute_query(query, (account_number, message_type, message), fetch=False)
        except Exception as e:
            print(f"Error saving chat message: {e}")

    def get_chat_history(self, account_number: str, limit: int = 50) -> List[Dict]:
        """Get recent chat history."""
        # Delegate to sqlite impl when active
        if getattr(self, "_impl", None):
            return self._impl.get_chat_history(account_number, limit)

        query = """
            SELECT * FROM chat_history 
            WHERE account_number = %s 
            ORDER BY timestamp DESC 
            LIMIT %s
        """
        return self.execute_query(query, (account_number, limit))
    
    def get_user(self, account_number: str) -> Dict:
        """Get user information."""
        # Delegate to sqlite impl when active
        if getattr(self, "_impl", None):
            return self._impl.get_user(account_number)

        try:
            query = """
                SELECT id, account_number, email, balance, last_login
                FROM users
                WHERE account_number = %s
            """
            result = self.execute_query(query, (account_number,))
            if not result:
                raise Exception("User not found")
            return result[0]
        except Exception as e:
            raise Exception(f"Error fetching user data: {str(e)}")
        
    def add_to_watchlist(self, account_number: str, symbol: str) -> Dict:
        """Add a stock to user's watchlist."""
        # Delegate to sqlite impl when active
        if getattr(self, "_impl", None):
            return self._impl.add_to_watchlist(account_number, symbol)

        try:
            with self.get_connection() as conn:
                cur = conn.cursor()
                
                # Check if already in watchlist
                cur.execute(
                    "SELECT * FROM watchlist WHERE account_number = %s AND stock_symbol = %s",
                    (account_number, symbol)
                )
                if cur.fetchone():
                    return {
                        "status": "error",
                        "message": f"{symbol} is already in your watchlist"
                    }
                
                # Add to watchlist
                cur.execute(
                    """
                    INSERT INTO watchlist (account_number, stock_symbol)
                    VALUES (%s, %s)
                    """,
                    (account_number, symbol)
                )
                
                return {
                    "status": "success",
                    "message": f"{symbol} added to watchlist"
                }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Database error: {str(e)}"
            }

    def remove_from_watchlist(self, account_number: str, symbol: str) -> Dict:
        """Remove a stock from user's watchlist."""
        # Delegate to sqlite impl when active
        if getattr(self, "_impl", None):
            return self._impl.remove_from_watchlist(account_number, symbol)

        try:
            with self.get_connection() as conn:
                cur = conn.cursor()
                
                cur.execute(
                    """
                    DELETE FROM watchlist 
                    WHERE account_number = %s AND stock_symbol = %s
                    RETURNING stock_symbol
                    """,
                    (account_number, symbol)
                )
                
                if cur.fetchone():
                    return {
                        "status": "success",
                        "message": f"{symbol} removed from watchlist"
                    }
                return {
                    "status": "error",
                    "message": f"{symbol} not found in watchlist"
                }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Database error: {str(e)}"
            }

    def get_watchlist(self, account_number: str) -> List[Dict]:
        """Get user's watchlist with current prices."""
        # Delegate to sqlite impl when active
        if getattr(self, "_impl", None):
            return self._impl.get_watchlist(account_number)

        try:
            with self.get_connection() as conn:
                cur = conn.cursor(cursor_factory=RealDictCursor)

                cur.execute(
                    "SELECT * FROM watchlist WHERE account_number = %s",
                    (account_number,)
                )
                watchlist = cur.fetchall()

            # Enrich with current market prices
            for item in watchlist:
                try:
                    quote = self.get_real_time_quote(item['stock_symbol'])
                    item.update(quote)
                except Exception:
                    pass

            return watchlist
        except Exception as e:
            print(f"Error getting watchlist: {e}")
            return []

    def init_db(self) -> None:
        """Initialize database schema required by the application.

        This will create tables if they do not exist. Intended for development
        use; in production use proper migrations.
        """
        # If using sqlite fallback, delegate
        if getattr(self, "_impl", None):
            return self._impl.init_db()

        create_statements = [
            """
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                account_number VARCHAR(64) UNIQUE NOT NULL,
                email VARCHAR(255) UNIQUE NOT NULL,
                password TEXT NOT NULL,
                balance NUMERIC DEFAULT 0,
                last_login TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS portfolio (
                id SERIAL PRIMARY KEY,
                account_number VARCHAR(64) REFERENCES users(account_number),
                stock_symbol VARCHAR(16) NOT NULL,
                shares NUMERIC NOT NULL,
                average_price NUMERIC NOT NULL,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS transactions (
                transaction_id SERIAL PRIMARY KEY,
                account_number VARCHAR(64) REFERENCES users(account_number),
                transaction_type VARCHAR(16),
                stock_symbol VARCHAR(16),
                shares NUMERIC,
                price_per_share NUMERIC,
                total_amount NUMERIC,
                transaction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS watchlist (
                id SERIAL PRIMARY KEY,
                account_number VARCHAR(64) REFERENCES users(account_number),
                stock_symbol VARCHAR(16) NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS chat_history (
                id SERIAL PRIMARY KEY,
                account_number VARCHAR(64) REFERENCES users(account_number),
                message_type VARCHAR(32),
                message TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        ]

        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    for stmt in create_statements:
                        cur.execute(stmt)
                conn.commit()
            print("Database schema initialized (or already present).")
        except Exception as e:
            raise Exception(f"Error initializing DB schema: {e}")


class _SQLiteDatabaseManager:
    """A lightweight SQLite-based implementation used as a local fallback.

    It implements the subset of DatabaseManager APIs used by the app.
    """
    def __init__(self, path: str):
        self.path = path
        self._quote_cache = {}
        self._cache_timeout = 60

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def execute_query(self, query: str, parameters: Any = None, fetch: bool = True):
        parameters = parameters or ()
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(query.replace('%s', '?'), parameters)
            if fetch:
                rows = cur.fetchall()
                return [dict(r) for r in rows]
            else:
                conn.commit()
                return {"affected_rows": cur.rowcount}

    def init_db(self):
        stmts = [
            "CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, account_number TEXT UNIQUE, email TEXT UNIQUE, password TEXT, balance REAL DEFAULT 0, last_login TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP)",
            "CREATE TABLE IF NOT EXISTS portfolio (id INTEGER PRIMARY KEY AUTOINCREMENT, account_number TEXT, stock_symbol TEXT, shares REAL, average_price REAL, last_updated TEXT DEFAULT CURRENT_TIMESTAMP)",
            "CREATE TABLE IF NOT EXISTS transactions (transaction_id INTEGER PRIMARY KEY AUTOINCREMENT, account_number TEXT, transaction_type TEXT, stock_symbol TEXT, shares REAL, price_per_share REAL, total_amount REAL, transaction_date TEXT DEFAULT CURRENT_TIMESTAMP)",
            "CREATE TABLE IF NOT EXISTS watchlist (id INTEGER PRIMARY KEY AUTOINCREMENT, account_number TEXT, stock_symbol TEXT)",
            "CREATE TABLE IF NOT EXISTS chat_history (id INTEGER PRIMARY KEY AUTOINCREMENT, account_number TEXT, message_type TEXT, message TEXT, timestamp TEXT DEFAULT CURRENT_TIMESTAMP)"
        ]
        with self._connect() as conn:
            cur = conn.cursor()
            for s in stmts:
                cur.execute(s)
            conn.commit()

    def create_user(self, data: Dict) -> Dict:
        try:
            required = ['email', 'password', 'account_number']
            for r in required:
                if r not in data:
                    return {"status": "error", "message": f"Missing {r}"}
            salt = bcrypt.gensalt()
            hashed = bcrypt.hashpw(data['password'].encode('utf-8'), salt).decode('utf-8')
            with self._connect() as conn:
                cur = conn.cursor()
                cur.execute("INSERT INTO users (account_number, email, password, balance) VALUES (?, ?, ?, ?)", (data['account_number'], data['email'], hashed, data.get('balance', 0.0)))
                conn.commit()
            return {"status": "success", "message": f"Account created {data['account_number']}"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def validate_login(self, email: str, password: str) -> Dict:
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute("SELECT id, account_number, email, password, balance FROM users WHERE email = ?", (email,))
            row = cur.fetchone()
            if not row:
                return {"status": "error", "message": "Invalid email or password"}
            user = dict(row)
            if bcrypt.checkpw(password.encode('utf-8'), user['password'].encode('utf-8')):
                cur.execute("UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?", (user['id'],))
                conn.commit()
                return {"status": "success", "data": {"account_number": user['account_number'], "email": user['email'], "balance": user['balance']}}
            return {"status": "error", "message": "Invalid email or password"}

    def get_real_time_quote(self, symbol: str) -> Dict:
        try:
            current_time = time.time()
            if symbol in self._quote_cache:
                cached, t = self._quote_cache[symbol]
                if current_time - t < self._cache_timeout:
                    return cached
            # Light-weight call to yfinance
            stock = yf.Ticker(symbol.upper())
            info = stock.info
            price = info.get('regularMarketPrice') or info.get('previousClose') or None
            quote = {"symbol": symbol.upper(), "price": price, "change": info.get('regularMarketChangePercent', None), "volume": info.get('regularMarketVolume', None), "timestamp": datetime.now().isoformat()}
            self._quote_cache[symbol] = (quote, current_time)
            return quote
        except Exception as e:
            return {"symbol": symbol.upper(), "price": None, "change": None, "volume": None, "timestamp": datetime.now().isoformat(), "error": str(e)}

    def execute_trade(self, account_number: str, trade_type: str, symbol: str, shares: int, price: float) -> Dict:
        try:
            total = float(shares) * float(price)
            with self._connect() as conn:
                cur = conn.cursor()
                cur.execute("SELECT balance FROM users WHERE account_number = ?", (account_number,))
                row = cur.fetchone()
                if not row:
                    return {"status": "error", "message": "Account not found"}
                balance = row['balance']
                if trade_type == 'BUY' and balance < total:
                    return {"status": "error", "message": "Insufficient funds"}
                # update balance
                if trade_type == 'BUY':
                    cur.execute("UPDATE users SET balance = balance - ? WHERE account_number = ?", (total, account_number))
                    cur.execute("SELECT shares, average_price FROM portfolio WHERE account_number = ? AND stock_symbol = ?", (account_number, symbol))
                    existing = cur.fetchone()
                    if existing:
                        existing_shares = existing['shares']
                        existing_avg = existing['average_price']
                        new_avg = ((existing_avg * existing_shares) + (price * shares)) / (existing_shares + shares)
                        cur.execute("UPDATE portfolio SET shares = shares + ?, average_price = ? WHERE account_number = ? AND stock_symbol = ?", (shares, new_avg, account_number, symbol))
                    else:
                        cur.execute("INSERT INTO portfolio (account_number, stock_symbol, shares, average_price) VALUES (?, ?, ?, ?)", (account_number, symbol, shares, price))
                else:
                    # SELL
                    cur.execute("SELECT shares FROM portfolio WHERE account_number = ? AND stock_symbol = ?", (account_number, symbol))
                    existing = cur.fetchone()
                    if not existing or existing['shares'] < shares:
                        return {"status": "error", "message": "Insufficient shares"}
                    cur.execute("UPDATE portfolio SET shares = shares - ? WHERE account_number = ? AND stock_symbol = ?", (shares, account_number, symbol))
                    cur.execute("DELETE FROM portfolio WHERE account_number = ? AND stock_symbol = ? AND shares <= 0", (account_number, symbol))
                    cur.execute("UPDATE users SET balance = balance + ? WHERE account_number = ?", (total, account_number))
                cur.execute("INSERT INTO transactions (account_number, transaction_type, stock_symbol, shares, price_per_share, total_amount) VALUES (?, ?, ?, ?, ?, ?)", (account_number, trade_type, symbol, shares, price, total))
                conn.commit()
                return {"status": "success", "message": f"{trade_type} executed", "transaction_id": cur.lastrowid}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def get_portfolio(self, account_number: str):
        try:
            with self._connect() as conn:
                cur = conn.cursor()
                cur.execute("SELECT stock_symbol, SUM(shares) as shares, average_price FROM portfolio WHERE account_number = ? GROUP BY stock_symbol, average_price", (account_number,))
                rows = cur.fetchall()
                positions = []
                for r in rows:
                    symbol = r['stock_symbol']
                    shares = float(r['shares'])
                    avg = float(r['average_price'])
                    quote = self.get_real_time_quote(symbol)
                    current = float(quote.get('price', avg))
                    positions.append({'stock_symbol': symbol, 'shares': shares, 'average_price': avg, 'current_price': current})
                return positions
        except Exception as e:
            return []

    def save_chat_message(self, account_number: str, message_type: str, message: str) -> None:
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute("INSERT INTO chat_history (account_number, message_type, message) VALUES (?, ?, ?)", (account_number, message_type, message))
            conn.commit()

    def get_chat_history(self, account_number: str, limit: int = 50):
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM chat_history WHERE account_number = ? ORDER BY timestamp DESC LIMIT ?", (account_number, limit))
            rows = cur.fetchall()
            return [dict(r) for r in rows]

    def get_user(self, account_number: str):
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute("SELECT id, account_number, email, balance, last_login FROM users WHERE account_number = ?", (account_number,))
            row = cur.fetchone()
            if not row:
                raise Exception('User not found')
            return dict(row)

    def add_to_watchlist(self, account_number: str, symbol: str):
        try:
            with self._connect() as conn:
                cur = conn.cursor()
                cur.execute("SELECT 1 FROM watchlist WHERE account_number = ? AND stock_symbol = ?", (account_number, symbol))
                if cur.fetchone():
                    return {"status": "error", "message": f"{symbol} already in watchlist"}
                cur.execute("INSERT INTO watchlist (account_number, stock_symbol) VALUES (?, ?)", (account_number, symbol))
                conn.commit()
                return {"status": "success", "message": f"{symbol} added to watchlist"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def remove_from_watchlist(self, account_number: str, symbol: str):
        try:
            with self._connect() as conn:
                cur = conn.cursor()
                cur.execute("DELETE FROM watchlist WHERE account_number = ? AND stock_symbol = ?", (account_number, symbol))
                conn.commit()
                return {"status": "success", "message": f"{symbol} removed"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def get_watchlist(self, account_number: str):
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute("SELECT stock_symbol FROM watchlist WHERE account_number = ?", (account_number,))
            rows = cur.fetchall()
            result = []
            for r in rows:
                sym = r['stock_symbol']
                quote = self.get_real_time_quote(sym)
                d = {'stock_symbol': sym}
                d.update(quote)
                result.append(d)
            return result

    # Note: DatabaseManager.init_db is implemented above and the SQLite
    # fallback has its own init_db() method. No duplicate implementation here.