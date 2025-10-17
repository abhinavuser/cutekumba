# Finance Agent (Local)

This project is a local financial agent that can answer questions, manage user accounts, and execute mock trades using a local LLM (Ollama) or a mock LLM for testing.

Quick start (recommended using SQLite fallback):

1. Create and activate virtualenv

```powershell
cd "e:\VS Code\LLMs\ai-project"
python -m venv .venv
.venv\Scripts\Activate.ps1
```

2. Install requirements

```powershell
pip install -r requirements.txt
```

3. Initialize DB (will create `ai_project_data.sqlite3` by default)

```powershell
python run_init_db.py
```

4. Run the smoke test

```powershell
python run_smoke_test.py
```

5. Run the Streamlit UI

```powershell
streamlit run app.py
```

Ollama
------
If you have Ollama installed locally and running at `http://localhost:11434` you can enable it in the Streamlit UI checkbox. You can see installed models by running `ollama list` in a terminal.

Inspecting the DB
-----------------
The SQLite file is `ai_project_data.sqlite3`. You can inspect it with `sqlite3` in PowerShell or use a GUI tool like DB Browser for SQLite.

Security & Notes
----------------
- This is a demo. Do not use the mock LLM or SQLite for production.
- Review the code before connecting real brokerage or executing real trades.
