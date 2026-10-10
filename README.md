# TalentGuard - Attrition Risk Dashboard

## Run locally
    pip install -r requirements.txt
    streamlit run dashboard.py

## Deploy free (Streamlit Community Cloud)
1. Push this folder to GitHub (keep data/ with the CSV). Never commit secrets.toml.
2. share.streamlit.io -> New app -> main file `dashboard.py`.
3. (Optional, for AI copilot) App settings -> Secrets -> add ANTHROPIC_API_KEY (see .streamlit/secrets.toml.example).
4. First load trains the model (~20s), then it is cached.

## Tabs
Overview | Watch list (+PDF briefs) | Employee deep-dive (+what-if, PDF) | Cycle planner |
Teams | Action tracker | Ask HR Copilot (rules, or Claude if key is set)

## Files
core.py (model, drivers, recommendations, planner, team stats) - dashboard.py (UI) - llm.py (Claude)
reports.py (PDF briefs) - tracker.py (action log) - data/ (CSV)

IMPORTANT: never name a file streamlit.py - it shadows the real package.
Action log: stored in data/action_log.csv; on Streamlit Cloud it resets on restart, so use the backup button.
