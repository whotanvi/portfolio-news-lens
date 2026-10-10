# Portfolio News Lens
Upload your holdings; the app fetches today's market headlines and uses Claude to trace each one
(directly and indirectly) to your portfolio, rating impact Low / Medium / High for today, this month
and a year or more. Impact = how hard the news hits the business x how much of your portfolio sits there.
No price forecasts. Not investment advice.

## Run locally
```
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
mkdir -p .streamlit && cp secrets.toml.example .streamlit/secrets.toml   # then paste your key
streamlit run app.py
```
Your API key lives only in `.streamlit/secrets.toml` (git-ignored). Never commit it or real holdings.

## CSV format
`symbol,quantity,buy_price` (buy_price optional). See `sample_holdings.csv`.

## Files
`app.py` the app | `dashboard_template.html` the dashboard | `demo_data.json` saved demo
