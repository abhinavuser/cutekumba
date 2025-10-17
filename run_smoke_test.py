from database.database_manager import DatabaseManager

# Initialize DB and run a brief smoke test: create user, validate login

def run():
    db = DatabaseManager()
    db.init_db()
    print('DB init done')

    res = db.create_user({'email': 'test@local', 'password': 'pass123', 'account_number': 'acct123'})
    print('create_user ->', res)

    login = db.validate_login('test@local', 'pass123')
    print('validate_login ->', login)

if __name__ == '__main__':
    run()
