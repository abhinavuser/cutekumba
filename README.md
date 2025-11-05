# Finance Agent 

This project is a financial agent that can answer questions, manage user accounts, and execute mock trades using a local LLM (Ollama) or a mock LLM for testing.

Quick start:

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

4. Run the agent

```powershell
python run_agent.py
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

