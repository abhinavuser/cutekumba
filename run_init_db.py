from database.database_manager import DatabaseManager
import sys

try:
    db = DatabaseManager()
    db.init_db()
    print("INIT_OK")
except Exception as e:
    print("INIT_ERROR:", e)
    sys.exit(0)
