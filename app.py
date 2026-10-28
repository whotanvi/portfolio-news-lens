"""Portfolio News Lens
Upload holdings -> fetch today's news -> Claude traces each headline to your
holdings -> dashboard shows impact (Low/Medium/High) for today, this month, a year+.
"""
import json
import os
import re

import feedparser
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
DEFAULT_MODEL = "gemini-2.5-flash"   # override with GEMINI_MODEL in secrets if Google renames it
MAX_RUNS_PER_SESSION = 5       # keeps one visitor from using up the free daily quota
HERE = os.path.dirname(os.path.abspath(__file__))

# News feeds (headline + link only; we never republish article text)
FEEDS = [
    ("Economic Times", "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms"),
    ("Mint", "https://www.livemint.com/rss/markets"),
    ("Business Standard", "https://www.business-standard.com/rss/markets-106.rss"),
]

st.set_page_config(page_title="Portfolio News Lens", page_icon="🔎", layout="wide")


def read_secret(name):
    """Read a secret from Streamlit secrets or environment, never from code."""
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.environ.get(name)


def load_json(name):
    with open(os.path.join(HERE, name), encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def fetch_news():
    """Fetch headlines from RSS. Falls back to a saved sample if feeds fail."""
    items, seen = [], set()
    for source, url in FEEDS:
        try:
            feed = feedparser.parse(url)
            for e in feed.entries[:10]:
                title = (e.get("title") or "").strip()
                if title and title.lower() not in seen:
                    seen.add(title.lower())
                    items.append({"title": title, "source": source, "link": e.get("link", "")})
        except Exception:
            continue
    if items:
        items = items[:30]
        try:  # save a copy so the demo still works if feeds fail later
            with open(os.path.join(HERE, "news_cache.json"), "w", encoding="utf-8") as f:
                json.dump(items, f)
        except Exception:
            pass
        return items, "live"
    for name in ("news_cache.json", "sample_news.json"):
        try:
            return load_json(name), "saved"
        except Exception:
            continue
    return [], "none"


def last_price(symbol):
    """Try to get the latest price for an NSE symbol. Returns None on failure."""
    try:
        import yfinance as yf
        p = yf.Ticker(symbol.upper() + ".NS").fast_info["last_price"]
        return float(p) if p else None
    except Exception:
        return None


def build_holdings(df):
    """Turn the uploaded table into a list of (symbol, weight in %)."""
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    if "symbol" not in df or "quantity" not in df:
        raise ValueError("Your file needs at least two columns: symbol and quantity.")
    df = df.dropna(subset=["symbol", "quantity"])
    df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df = df.dropna(subset=["quantity"])
    df = df[df["quantity"] > 0].head(25)
    if df.empty:
        raise ValueError("No valid rows found. Check symbol and quantity.")
    prices = pd.to_numeric(df["buy_price"], errors="coerce") if "buy_price" in df else pd.Series([None] * len(df), index=df.index)
    values = []
    for (_, row), p in zip(df.iterrows(), prices):
        if pd.isna(p):
            p = last_price(row["symbol"])
        values.append(row["quantity"] * (p if p else 1))
    total = sum(values)
    return [(s, round(v / total * 100, 1)) for s, v in zip(df["symbol"], values)]


SYSTEM = (
    "You are a careful equity research assistant for Indian markets. "
    "You explain how news affects a specific portfolio, directly and indirectly, "
    "with clear cause-and-effect reasoning. You never give price targets or numbers "
    "forecasts. Reply with JSON only, no other text."
)


def analyze(holdings, headlines, key):
    from openai import OpenAI   # Gemini speaks the same format as OpenAI
    client = OpenAI(api_key=key, base_url=GEMINI_URL)
    hold_txt = "\n".join(f"{i}. {s} ({w}% of portfolio)" for i, (s, w) in enumerate(holdings))
    news_txt = "\n".join(f"{i}. {h['title']}" for i, h in enumerate(headlines))
    prompt = f"""HOLDINGS (index. NSE symbol, weight):
{hold_txt}

HEADLINES (index. title):
{news_txt}

Task:
1. Give each holding a short sector name (e.g. Autos, Banks, IT, Paints, Energy, Power, Pharma, FMCG, Metals).
2. Pick up to 8 headlines that genuinely affect at least one holding, directly or INDIRECTLY (through input costs, currency, rates, demand, etc.). Ignore the rest.
3. For each, give a 3-step cause chain, a plain-English 'why' (max 2 sentences), and effects per affected holding.

Effect values are three integers [today, this month, a year or more], each from -3 to 3.
Sign = direction (negative = bad for the holding, positive = good). Size = how hard the news hits the business (1 mild, 2 solid, 3 severe). Judge each headline on its own. The year-or-more value is a long-term thesis, not a forecast. Do not consider portfolio weight; the app does that.

Return JSON exactly like:
{{"sectors": {{"SYMBOL": "Sector"}},
 "news": [{{"headline_index": 0, "chain": ["step 1", "step 2", "step 3"], "why": "...", "effects": {{"<holding index>": [0, 0, 0]}}}}]}}"""
    resp = client.chat.completions.create(
        model=read_secret("GEMINI_MODEL") or DEFAULT_MODEL,
        max_tokens=8000,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": prompt}],
    )
    text = resp.choices[0].message.content or ""
    data = json.loads(text[text.index("{"): text.rindex("}") + 1])
    H = [[s, data.get("sectors", {}).get(s, "Other"), w] for s, w in holdings]
    NEWS = []
    for item in data.get("news", []):
        idx = item.get("headline_index")
        if not isinstance(idx, int) or not 0 <= idx < len(headlines):
            continue
        eff = {}
        for k, v in (item.get("effects") or {}).items():
            try:
                k = int(k)
                if 0 <= k < len(holdings) and len(v) == 3:
                    eff[k] = [max(-3, min(3, int(x))) for x in v]
            except Exception:
                continue
        if eff:
            NEWS.append({"t": headlines[idx]["title"], "src": headlines[idx]["source"],
                         "chain": (item.get("chain") or [])[:4],
                         "why": item.get("why", ""), "e": eff})
    return H, NEWS


def render(H, NEWS, tag):
    with open(os.path.join(HERE, "dashboard_template.html"), encoding="utf-8") as f:
        html = f.read()
    safe = lambda obj: json.dumps(obj).replace("</", "<\\/")
    html = html.replace("__H__", safe(H)).replace("__NEWS__", safe(NEWS)).replace("__TAG__", tag)
    components.html(html, height=2300, scrolling=True)


# ---------------- page ----------------
st.title("Portfolio News Lens")
st.caption("Add your holdings, and see what today's news means for what you own.")
st.info("**Not investment advice.** This is an educational tool that reads news, not a recommendation "
        "to buy or sell anything. Your holdings are sent to Google's Gemini API to produce the analysis, "
        "and free-tier inputs may be used by Google to improve its products. "
        "Please don't upload anything you wouldn't want processed there.")

# Optional access code for the public link
code = read_secret("APP_ACCESS_CODE")
if code and st.text_input("Access code", type="password") != code:
    st.info("Enter the access code to continue.")
    st.stop()

mode = st.radio("How do you want to add holdings?",
                ["Upload CSV", "Add manually", "Use sample portfolio"], horizontal=True)

df = None
if mode == "Upload CSV":
    st.caption("Columns: symbol, quantity, and optionally buy_price (NSE symbols like TCS, HDFCBANK).")
    up = st.file_uploader("Holdings CSV", type="csv")
    if up:
        df = pd.read_csv(up)
elif mode == "Add manually":
    df = st.data_editor(pd.DataFrame({"symbol": [""], "quantity": [0], "buy_price": [0.0]}),
                        num_rows="dynamic", use_container_width=True)
else:
    df = pd.read_csv(os.path.join(HERE, "sample_holdings.csv"))
    st.dataframe(df, use_container_width=True)

col1, col2 = st.columns(2)
go = col1.button("Analyse my portfolio", type="primary")
demo = col2.button("Show saved demo (no API call)")

if demo:
    d = load_json("demo_data.json")
    render(d["H"], d["NEWS"], "Saved sample, not real data")
elif go:
    key = read_secret("GEMINI_API_KEY")
    st.session_state["runs"] = st.session_state.get("runs", 0) + 1
    if df is None:
        st.error("Add some holdings first.")
    elif not key:
        st.error("No API key found. Add GEMINI_API_KEY to your secrets.")
    elif st.session_state["runs"] > MAX_RUNS_PER_SESSION:
        st.warning("Run limit reached for this session.")
    else:
        try:
            holdings = build_holdings(df)
            with st.spinner("Fetching today's news..."):
                headlines, how = fetch_news()
            if not headlines:
                st.error("Could not get any headlines.")
            else:
                with st.spinner("Claude is tracing the news to your holdings (about a minute)..."):
                    H, NEWS = analyze(holdings, headlines, key)
                if not NEWS:
                    st.info("None of today's headlines clearly affect these holdings.")
                render(H, NEWS, f"Your portfolio, {len(headlines)} headlines ({how})")
        except Exception as e:
            msg = str(e).lower()
            if "429" in msg or "quota" in msg or "rate" in msg:
                st.warning("The free daily limit has been reached. Please try again later, "
                           "or use the saved demo button.")
            else:
                st.error(f"Something went wrong: {e}")
