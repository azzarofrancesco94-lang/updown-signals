# app.py
import re
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

# =========================
# CONFIG (prima di ogni altro st.*)
# =========================
st.set_page_config(page_title="UpDown Signals", layout="wide")

st.title("📊 UpDown Signals")
st.write("Il tuo assistente per segnali di trading basati su analisi tecnica e fondamentale")
st.caption("⚠️ Educational only – non è consulenza finanziaria.")

# =========================
# UTILS
# =========================
def safe_ticker_info(t):
    """Restituisce un dict info sempre valido (evita crash se yfinance fallisce)."""
    try:
        d = t.info
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}

def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Se le colonne sono MultiIndex, le appiattisce in stringhe leggibili."""
    if isinstance(df.columns, pd.MultiIndex):
        flat_cols = []
        for col in df.columns:
            parts = [str(x) for x in col if x not in (None, "", " ")]
            flat_cols.append("_".join(parts) if parts else "col")
        df = df.copy()
        df.columns = flat_cols
    return df

def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Restituisce un DataFrame con colonne canonicali: Open, High, Low, Close, Volume (+Adj Close se disponibile).
    Riconosce i nomi indipendentemente dalla posizione del ticker (es. 'AAPL_Close' o 'Close_AAPL').
    Se 'Close' manca ma esiste 'Adj Close', usa quello come Close.
    """
    if df is None or df.empty:
        return df

    df = flatten_columns(df)

    def find_first(pattern_fn):
        for c in df.columns:
            lc = c.lower()
            tokens = re.findall(r"[a-z]+", lc)  # tokenizza parole
            if pattern_fn(tokens):
                return c
        return None

    col_map = {}
    col_map["Open"]      = find_first(lambda t: "open" in t)
    col_map["High"]      = find_first(lambda t: "high" in t)
    col_map["Low"]       = find_first(lambda t: "low" in t)
    col_map["Close"]     = find_first(lambda t: ("close" in t) and ("adj" not in t))
    col_map["Adj Close"] = find_first(lambda t: ("close" in t) and ("adj" in t))
    col_map["Volume"]    = find_first(lambda t: "volume" in t)

    out = pd.DataFrame(index=df.index)
    for k, src in col_map.items():
        if src is not None:
            out[k] = df[src].astype(float)

    if "Close" not in out.columns and "Adj Close" in out.columns:
        out["Close"] = out["Adj Close"]

    return out

# =========================
# COSTANTI
# =========================
HEATMAP_TICKERS = ["AAPL","MSFT","TSLA","AMZN","GOOGL","META","NVDA","NFLX","BBVA"]

# =========================
# CACHE LAYER
# =========================
@st.cache_data(ttl=300, show_spinner=False)  # 5 minuti
def build_heatmap_df(tickers: list) -> pd.DataFrame:
    """Crea il DataFrame per la heatmap (settoriale, change %, market cap)."""
    rows = []
    for t in tickers:
        try:
            stock = yf.Ticker(t)
            info = safe_ticker_info(stock)

            price = info.get("currentPrice", None)
            prev = info.get("previousClose", None)
            market_cap = info.get("marketCap", 1)
            sector = info.get("sector", "Other")

