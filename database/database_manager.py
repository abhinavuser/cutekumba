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

    def get_connection(self):
        """Create and return a database connection with dict cursor."""
        try:
            return psycopg2.connect(**self.db_config, cursor_factory=RealDictCursor)
        except Exception as e:
            raise Exception(f"Database connection error: {str(e)}")

    def execute_query(self, query: str, parameters: Any = None, fetch: bool = True) -> Optional[List[Dict]]:
        """Execute a database query with proper error handling and connection management."""
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
        conn = None
        try:
            print(f"Attempting to create user with email: {data['email']}")
            
            # Validate required fields
            required_fields = ['email', 'password', 'account_number']
            missing_fields = [field for field in required_fields if field not in data]
            if missing_fields:
                raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")

            # Hash password
            salt = bcrypt.gensalt()
            hashed_password = bcrypt.hashpw(data['password'].encode('utf-8'), salt)
            
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    # First check if user already exists
                    cur.execute("SELECT email FROM users WHERE email = %s", (data['email'],))
                    if cur.fetchone():
                        raise ValueError("User with this email already exists")

                    query = """
                        INSERT INTO users (account_number, email, password, balance)
                        VALUES (%s, %s, %s, %s)
                        RETURNING id, account_number, email, balance, created_at
                    """

                    cur.execute(query, (
                        data['account_number'],
                        data['email'],
                        hashed_password.decode('utf-8'),
                        data.get('balance', 0.00)
                    ))

                    result = cur.fetchone()
                    conn.commit()

                    return {
                        "status": "success",
                        "message": f"Account created successfully with number {result['account_number']}",
                        "data": result
                    }
        except Exception as e:
            if conn:
                conn.rollback()
            print(f"Error creating user: {str(e)}")
            return {"status": "error", "message": str(e)}
        finally:
            if conn:
                conn.close()

    def validate_login(self, email: str, password: str) -> Dict:
        """Validate user login credentials."""
        conn = None
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
            if conn:
                conn.rollback()
            print(f"Login error: {str(e)}")
            return {"status": "error", "message": str(e)}
        finally:
            if conn:
                conn.close()

    def get_real_time_quote(self, symbol: str) -> Dict:
            """Get real-time stock quote using yfinance with caching and rate limiting."""
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
                        "timestamp": datetime.now()
                    }
                    self._quote_cache[symbol] = (quote_data, current_time)
                    return quote_data

                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        stock = yf.Ticker(symbol.upper())
                        info = stock.info
                        
                        if 'regularMarketPrice' not in info:
                            raise ValueError(f"No price data available for {symbol}")
                        
                        quote_data = {
                            "symbol": symbol.upper(),
                            "price": info.get('regularMarketPrice', 0.0),
                            "change": info.get('regularMarketChangePercent', 0.0),
                            "volume": info.get('regularMarketVolume', 0),
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
                    
                # Return safe default with error indication
                return {
                    "symbol": symbol.upper(),
                    "price": 169.50,  # Default price for testing
                    "change": 0.0,
                    "volume": 0,
                    "timestamp": datetime.now().isoformat(),
                    "error": str(e)
                }

    def execute_trade(self, account_number: str, trade_type: str, 
                     symbol: str, shares: int, price: float) -> Dict:
        """Execute a stock trade with proper validation and error handling."""
        try:
            total_amount = float(shares) * float(price)
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    # Lock user's balance row
                    cur.execute("SELECT balance FROM users WHERE account_number = %s FOR UPDATE", (account_number,))
                    user = cur.fetchone()
                    if not user:
                        raise ValueError("Account not found")

                    # Calculate current holdings
                    cur.execute("SELECT SUM(shares) as total_shares FROM portfolio WHERE account_number = %s AND stock_symbol = %s", (account_number, symbol))
                    position = cur.fetchone()
                    current_shares = int(position['total_shares']) if position and position['total_shares'] else 0

                    if trade_type == 'SELL' and current_shares < shares:
                        raise ValueError(f"Insufficient shares for this trade. Required: {shares}, Available: {current_shares}")

                    current_balance = float(user['balance'])
                    if trade_type == 'BUY' and current_balance < total_amount:
                        raise ValueError(f"Insufficient funds for this trade. Required: ${total_amount:.2f}, Available: ${current_balance:.2f}")

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
            # Bubble up an error as structured response
            return {"status": "error", "message": str(e)}
                
    def get_portfolio(self, account_number: str) -> List[Dict]:
        """Get user's consolidated portfolio with current market values."""
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
        query = """
            SELECT * FROM chat_history 
            WHERE account_number = %s 
            ORDER BY timestamp DESC 
            LIMIT %s
        """
        return self.execute_query(query, (account_number, limit))
    
    def get_user(self, account_number: str) -> Dict:
        """Get user information."""
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