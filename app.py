# Write the generated Streamlit app with advanced watchlist to app.py
app_code = r'''# app.py
import re
import io
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# =========================
# CONFIG (prima di ogni altro st.*)
# =========================
st.set_page_config(page_title="UpDown Signals", layout="wide")

st.title("📊 UpDown Signals")
st.write("Il tuo assistente per segnali di trading basati su analisi tecnica e fondamentale")
st.caption("⚠️ Educational only – non è consulenza finanziaria.")

# =========================
# UTILS BASE
# =========================
def safe_ticker_info(t):
    """Ritorna un dict info sempre valido, evitando crash di yfinance."""
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

# ---------- Indicatori tecnici ----------
def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(gain, index=series.index).rolling(period).mean()
    roll_down = pd.Series(loss, index=series.index).rolling(period).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def macd(series: pd.Series, fast=12, slow=26, signal=9):
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def keltner_channels(df: pd.DataFrame, ema_period: int = 20, atr_period: int = 10, atr_mult: float = 2.0):
    mid = ema(df["Close"], ema_period)
    k_atr = atr(df, atr_period)
    upper = mid + atr_mult * k_atr
    lower = mid - atr_mult * k_atr
    return mid, upper, lower

# ---------- Backtest ----------
def backtest_from_signals(close: pd.Series,
                          signal_series: pd.Series,
                          allow_short: bool = False,
                          fee_bps: float = 0.0):
    """
    Backtest semplice daily:
    - positions: 1 per BUY, 0 HOLD (o -1 per SELL se allow_short)
    - esecuzione T+1 (shift delle posizioni)
    - costi: fee_bps applicato al cambio posizione (entry/exit/flip)
    """
    ret = close.pct_change().fillna(0.0)

    if allow_short:
        pos = np.where(signal_series == "BUY", 1,
                       np.where(signal_series == "SELL", -1, 0))
    else:
        pos = np.where(signal_series == "BUY", 1, 0)

    pos = pd.Series(pos, index=close.index).shift(1).fillna(0.0)

    pos_change = pos.diff().abs().fillna(pos.abs())
    fee = (fee_bps / 10_000.0) * pos_change

    strat_ret = (pos * ret) - fee
    equity = (1 + strat_ret).cumprod()
    high_watermark = equity.cummax()
    drawdown = (equity / high_watermark) - 1.0
    max_dd = drawdown.min()

    n = len(strat_ret)
    total_return = equity.iloc[-1] - 1.0
    cagr = (equity.iloc[-1]) ** (252.0 / max(n, 1)) - 1.0 if n > 0 else np.nan
    vol_ann = strat_ret.std() * np.sqrt(252) if strat_ret.std() > 0 else np.nan
    sharpe = (strat_ret.mean() * 252) / vol_ann if vol_ann and vol_ann > 0 else np.nan
    win_rate = (strat_ret[strat_ret > 0].count() / max(1, strat_ret.count())) * 100

    res = {
        "total_return": float(total_return),
        "cagr": float(cagr) if pd.notna(cagr) else np.nan,
        "vol_ann": float(vol_ann) if pd.notna(vol_ann) else np.nan,
        "sharpe": float(sharpe) if pd.notna(sharpe) else np.nan,
        "max_drawdown": float(max_dd) if pd.notna(max_dd) else np.nan,
        "win_rate": float(win_rate)
    }
    df_bt = pd.DataFrame({
        "Close": close,
        "Position": pos,
        "Strategy_Return": strat_ret,
        "Equity": equity,
        "Drawdown": drawdown
    })
    return res, df_bt

# ---------- Utils formattazione ----------
def fmt_pct(x, na="N/A", digits=2):
    return (f"{x*100:.{digits}f}%" if isinstance(x, (int, float)) and pd.notna(x) else na)

def fmt_num(x, na="N/A", digits=0):
    return (f"${x:,.{digits}f}" if isinstance(x, (int, float)) and pd.notna(x) else na)

def safe_get(d: dict, key: str, default=None):
    try:
        v = d.get(key, default)
        return v if v is not None else default
    except Exception:
        return default

# ---------- Fondamentali: estrazione (CACHE) ----------
@st.cache_data(ttl=600, show_spinner=False)  # 10 minuti
def get_fundamentals(ticker: str) -> dict:
    """
    Ritorna:
      - snapshot: KPI principali per Valuation, Profitability, Growth, Solidità, Dividendi
      - series: serie annuali/trimestrali (Revenue/Earnings) per grafici
      - criteria: dizionario di booleane per categoria -> {criterio: True/False}
    (Lo score viene calcolato separatamente in base ai pesi scelti nell’UI.)
    """
    t = yf.Ticker(ticker)
    info = {}
    try:
        info = t.info or {}
    except Exception:
        info = {}

    # ---- Valuation ----
    trailing_pe = safe_get(info, "trailingPE")
    forward_pe = safe_get(info, "forwardPE")
    peg = safe_get(info, "pegRatio")
    ps = safe_get(info, "priceToSalesTrailing12Months")
    pb = safe_get(info, "priceToBook")
    ev = safe_get(info, "enterpriseValue")
    ev_to_ebitda = safe_get(info, "enterpriseToEbitda")
    ev_to_rev = safe_get(info, "enterpriseToRevenue")
    ebitda = safe_get(info, "ebitda")

    if ev_to_ebitda is None and isinstance(ev, (int, float)) and isinstance(ebitda, (int, float)) and ebitda not in (0, None):
        ev_to_ebitda = ev / ebitda
    if ev_to_rev is None and isinstance(ev, (int, float)):
        revenue_ttm = safe_get(info, "totalRevenue")
        if isinstance(revenue_ttm, (int, float)) and revenue_ttm not in (0, None):
            ev_to_rev = ev / revenue_ttm

    # ---- Profitability ----
    gross_m = safe_get(info, "grossMargins")
    op_m = safe_get(info, "operatingMargins")
    net_m = safe_get(info, "profitMargins")
    roe = safe_get(info, "returnOnEquity")
    roa = safe_get(info, "returnOnAssets")

    # ---- Growth ----
    rev_growth = safe_get(info, "revenueGrowth")
    eps_q_growth = safe_get(info, "earningsQuarterlyGrowth")
    eps_ttm_growth = safe_get(info, "earningsGrowth")

    # Revenue CAGR da dati annuali
    revenue_cagr = None
    rev_series_yearly = None
    try:
        df_earn = t.earnings  # annuale: Revenue, Earnings
        if isinstance(df_earn, pd.DataFrame) and not df_earn.empty and "Revenue" in df_earn.columns:
            rev_series_yearly = df_earn["Revenue"].dropna()
            if len(rev_series_yearly) >= 3:
                first = float(rev_series_yearly.iloc[0]); last = float(rev_series_yearly.iloc[-1])
                years = len(rev_series_yearly) - 1
                if first > 0 and years > 0:
                    revenue_cagr = (last / first) ** (1 / years) - 1
    except Exception:
        pass

    # ---- Solidità ----
    debt_to_equity = safe_get(info, "debtToEquity")
    current_ratio = safe_get(info, "currentRatio")
    quick_ratio = safe_get(info, "quickRatio")
    total_debt = safe_get(info, "totalDebt")
    total_cash = safe_get(info, "totalCash")

    # ---- Dividendo ----
    dividend_yield = safe_get(info, "dividendYield")
    payout_ratio = safe_get(info, "payoutRatio")
    five_year_yield = safe_get(info, "fiveYearAvgDividendYield")

    # ---- Serie trimestrali per grafici ----
    rev_series_q = None
    earn_series_q = None
    try:
        qearn = t.quarterly_earnings  # Revenue, Earnings
        if isinstance(qearn, pd.DataFrame) and not qearn.empty:
            rev_series_q = qearn["Revenue"].dropna()
            earn_series_q = qearn["Earnings"].dropna()
    except Exception:
        pass

    snapshot = {
        "Valuation": {
            "Trailing P/E": trailing_pe,
            "Forward P/E": forward_pe,
            "PEG": peg,
            "P/S (TTM)": ps,
            "P/B": pb,
            "EV/EBITDA": ev_to_ebitda,
            "EV/Revenue": ev_to_rev
        },
        "Profitability": {
            "Gross Margin": gross_m,
            "Operating Margin": op_m,
            "Net Margin": net_m,
            "ROE": roe,
            "ROA": roa
        },
        "Growth": {
            "Revenue Growth (prov)": rev_growth,
            "EPS Growth QoQ": eps_q_growth,
            "EPS Growth TTM": eps_ttm_growth,
            "Revenue CAGR (annuale)": revenue_cagr
        },
        "Solidity": {
            "Debt/Equity": debt_to_equity,
            "Current Ratio": current_ratio,
            "Quick Ratio": quick_ratio,
            "Total Debt": total_debt,
            "Total Cash": total_cash
        },
        "Dividends": {
            "Dividend Yield": dividend_yield,
            "Payout Ratio": payout_ratio,
            "5Y Avg Yield": five_year_yield
        }
    }

    series = {
        "annual": {"Revenue": rev_series_yearly if isinstance(rev_series_yearly, pd.Series) else None},
        "quarterly": {
            "Revenue": rev_series_q if isinstance(rev_series_q, pd.Series) else None,
            "Earnings": earn_series_q if isinstance(earn_series_q, pd.Series) else None
        }
    }

    # ---- Criteri (True/False) per ciascuna categoria ----
    criteria = {
        "Profitability": {
            "ROE > 10%": isinstance(roe, (int, float)) and roe > 0.10,
            "Operating Margin > 12%": isinstance(op_m, (int, float)) and op_m > 0.12,
            "Net Margin > 10%": isinstance(net_m, (int, float)) and net_m > 0.10
        },
        "Growth": {
            "Revenue CAGR > 8%": isinstance(revenue_cagr, (int, float)) and revenue_cagr > 0.08,
            "EPS QoQ > 10%": isinstance(eps_q_growth, (int, float)) and eps_q_growth > 0.10,
            "EPS TTM > 8%": isinstance(eps_ttm_growth, (int, float)) and eps_ttm_growth > 0.08
        },
        "Valuation": {
            "Forward P/E < 25": isinstance(forward_pe, (int, float)) and forward_pe < 25,
            "PEG < 1.5": isinstance(peg, (int, float)) and peg < 1.5,
            "P/S < 6": isinstance(ps, (int, float)) and ps < 6
        },
        "Solidity": {
            "Debt/Equity < 1.0": isinstance(debt_to_equity, (int, float)) and debt_to_equity < 1.0,
            "Current Ratio > 1.2": isinstance(current_ratio, (int, float)) and current_ratio > 1.2
        },
        "Dividends": {
            "Yield > 1%": isinstance(dividend_yield, (int, float)) and dividend_yield > 0.01,
            "Payout < 80%": isinstance(payout_ratio, (int, float)) and payout_ratio < 0.80
        }
    }

    return {"snapshot": snapshot, "series": series, "criteria": criteria}

def compute_weighted_score(criteria: dict, weights_norm: dict) -> tuple[float, str, list, dict]:
    """
    Calcola: score (0..100), band, motivazioni (criteri soddisfatti), contributi per categoria.
    weights_norm deve sommare a 1.0 (normalizziamo comunque se necessario).
    """
    s = sum(max(0.0, float(v)) for v in weights_norm.values())
    wn = {k: (max(0.0, float(v)) / s if s > 0 else 0.0) for k, v in weights_norm.items()}

    total_score = 0.0
    reasons = []
    contributions = {}
    for cat, tests in criteria.items():
        tests_list = list(tests.values())
        n = len(tests_list) if len(tests_list) > 0 else 1
        passed = sum(1 for v in tests_list if v)
        cat_ratio = passed / n
        contrib = cat_ratio * wn.get(cat, 0.0) * 100.0
        contributions[cat] = contrib
        total_score += contrib
        for name, ok in tests.items():
            if ok:
                reasons.append(f"{cat}: {name}")

    band = "Strong" if total_score >= 70 else ("Neutral" if total_score >= 50 else "Weak")
    return total_score, band, reasons, contributions

# =========================
# WATCHLIST AVANZATA
# =========================
DEFAULT_WATCHLIST = [
    # Mega Cap USA
    "AAPL","MSFT","GOOGL","AMZN","META","NVDA","TSLA",
    # ETF
    "SPY","QQQ","IWM","DIA",
    # Bancari
    "JPM","BAC","C","WFC",
    # Energia
    "XOM","CVX","COP",
    # Europa (alcuni esempi)
    "RACE","SAP.DE","MBG.DE","NESN.SW","NOVN.SW",
    # Crypto
    "BTC-USD","ETH-USD"
]

PRESET_LISTS = {
    "Default MegaCaps": DEFAULT_WATCHLIST,
    "Tech & AI": [
        "AAPL","MSFT","GOOGL","META","NVDA","TSLA","AMD","AVGO","ADBE","CRM","SNOW","PLTR","SMCI"
    ],
    "Dow Jones 30 (subset)": [
        "AAPL","MSFT","CRM","MCD","V","JNJ","PG","HD","UNH","VZ","INTC","CSCO"
    ],
    "ETF & Crypto": ["SPY","QQQ","IWM","DIA","BTC-USD","ETH-USD","GLD","TLT"],
}

@st.cache_data(ttl=3600, show_spinner=False)
def load_sp500_from_wikipedia() -> list:
    """Prova a caricare la lista S&P500 da Wikipedia. Se fallisce, ritorna []."""
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        df = pd.read_html(url)[0]
        syms = df["Symbol"].astype(str).str.strip().tolist()
        # Alcuni simboli hanno "." come separatore su Yahoo -> converti a '-'
        syms = [s.replace(".", "-") for s in syms]
        return syms
    except Exception:
        return []

@st.cache_data(ttl=3600, show_spinner=False)
def load_nasdaq100_from_wikipedia() -> list:
    try:
        url = "https://en.wikipedia.org/wiki/Nasdaq-100"
        tables = pd.read_html(url)
        # In genere la 1a o 2a tabella contiene i componenti
        for df in tables:
            if "Ticker" in df.columns or "Symbol" in df.columns:
                col = "Ticker" if "Ticker" in df.columns else "Symbol"
                syms = df[col].astype(str).str.strip().tolist()
                syms = [s.replace(".", "-") for s in syms]
                # Filtro simboli strani
                syms = [s for s in syms if len(s) > 0 and len(s) <= 10]
                if len(syms) >= 50:
                    return syms
        return []
    except Exception:
        return []

def normalize_ticker_list(raw_list: list[str]) -> list[str]:
    seen = set()
    cleaned = []
    for t in raw_list:
        t2 = t.strip().upper()
        if not t2:
            continue
        # Consenti lettere, numeri, trattino, punto
        if re.fullmatch(r"[A-Z0-9\-\.]+", t2) is None:
            continue
        if t2 not in seen:
            seen.add(t2)
            cleaned.append(t2)
    return cleaned

# =========================
# CACHE LAYER (dati prezzo e heatmap)
# =========================
@st.cache_data(ttl=300, show_spinner=False)  # 5 minuti
def build_heatmap_df(tickers: list) -> pd.DataFrame:
    rows = []
    for t in tickers:
        try:
            stock = yf.Ticker(t)
            info = safe_ticker_info(stock)
            price = info.get("currentPrice", None)
            prev = info.get("previousClose", None)
            market_cap = info.get("marketCap", 1)
            sector = info.get("sector", "Other")
            if isinstance(price, (int, float)) and isinstance(prev, (int, float)) and prev not in (0, None):
                change = (price - prev) / prev
                rows.append({
                    "Sector": sector or "Other",
                    "Ticker": t,
                    "Change": change,
                    "MarketCap": float(market_cap) if market_cap else 1.0
                })
        except Exception:
            continue
    return pd.DataFrame(rows)

@st.cache_data(ttl=120, show_spinner=False)  # 2 minuti
def get_history_normalized(ticker: str, period: str) -> pd.DataFrame:
    raw = yf.download(ticker, period=period, progress=False, auto_adjust=False, group_by="column")
    if raw is None or raw.empty:
        return pd.DataFrame()
    return normalize_ohlcv(raw)

# =========================
# LAYOUT: HEATMAP + CONTROLLI AFFIANCATI (con Watchlist avanzata)
# =========================
left, right = st.columns([3, 1], gap="large")

with right:
    st.subheader("⚙️ Impostazioni")

    # --- Watchlist avanzata ---
    st.markdown("**📋 Watchlist avanzata**")
    if "watchlist" not in st.session_state:
        st.session_state["watchlist"] = DEFAULT_WATCHLIST

    preset = st.selectbox("Preset watchlist", list(PRESET_LISTS.keys()) + ["S&P 500 (Wikipedia)", "Nasdaq-100 (Wikipedia)", "Custom"], index=0)

    custom_area = st.text_area(
        "Ticker personalizzati (separati da virgola)",
        value=",".join(st.session_state["watchlist"]),
        help="Esempio: AAPL, MSFT, NVDA, SPY, BTC-USD"
    )

    uploaded = st.file_uploader("(Opzionale) Carica CSV con colonna 'Symbol' o prima colonna", type=["csv"]) 

    col_w1, col_w2 = st.columns(2)
    with col_w1:
        include_etf = st.checkbox("+ ETF comuni", value=True)
        include_crypto = st.checkbox("+ Crypto (BTC/ETH)", value=True)
    with col_w2:
        heat_max = st.slider("Max titoli in heatmap", min_value=10, max_value=200, value=60, step=10)
        
    apply_watch = st.button("📥 Applica watchlist")

    # Costruzione watchlist
    final_list = []
    if preset in PRESET_LISTS:
        final_list.extend(PRESET_LISTS[preset])
    elif preset == "S&P 500 (Wikipedia)":
        final_list.extend(load_sp500_from_wikipedia())
    elif preset == "Nasdaq-100 (Wikipedia)":
        final_list.extend(load_nasdaq100_from_wikipedia())

    # Aggiungi custom da textarea
    if custom_area:
        user_list = [x.strip() for x in custom_area.split(",")]
        final_list.extend(user_list)

    # File CSV
    if uploaded is not None:
        try:
            df_up = pd.read_csv(uploaded)
            if "Symbol" in df_up.columns:
                final_list.extend(df_up["Symbol"].astype(str).tolist())
            else:
                # usa la prima colonna
                first_col = df_up.columns[0]
                final_list.extend(df_up[first_col].astype(str).tolist())
        except Exception as e:
            st.warning(f"CSV non letto: {e}")

    # Extra inclusioni rapide
    if include_etf:
        final_list.extend(["SPY","QQQ","IWM","DIA","XLK","XLF","XLE","XLI","XLV","XLY","XLP","XLB","XLU"])
    if include_crypto:
        final_list.extend(["BTC-USD","ETH-USD"]) 

    # Normalizza e limita per heatmap
    final_list = normalize_ticker_list(final_list)
    if len(final_list) == 0:
        final_list = DEFAULT_WATCHLIST

    if apply_watch:
        st.session_state["watchlist"] = final_list
        st.success(f"Watchlist aggiornata: {len(final_list)} simboli")

    # Select box del titolo
    ticker = st.selectbox("Seleziona un Titolo", st.session_state["watchlist"], index=0)
    period = st.selectbox("Periodo", ["3mo","6mo","1y"], index=0)

    # Parametri tecnici
    with st.expander("Parametri Tecnici", expanded=True):
        ema_fast = st.number_input("EMA Fast", min_value=5, max_value=100, value=20, step=1)
        ema_slow = st.number_input("EMA Slow", min_value=ema_fast+1, max_value=200, value=50, step=1,
                                   help="Deve essere > EMA Fast")
        atr_period = st.number_input("ATR Period (risk)", min_value=5, max_value=50, value=14, step=1)
        keltner_atr_period = st.number_input("ATR Period (Keltner)", min_value=5, max_value=50, value=10, step=1)
        keltner_mult = st.slider("Keltner Multiplier", min_value=1.0, max_value=3.0, value=2.0, step=0.1)
        use_rsi = st.checkbox("Usa filtro RSI (>50 per BUY, <50 per SELL)", value=True)
        macd_fast = st.number_input("MACD Fast", min_value=5, max_value=30, value=12, step=1)
        macd_slow = st.number_input("MACD Slow", min_value=10, max_value=60, value=26, step=1)
        macd_signal = st.number_input("MACD Signal", min_value=5, max_value=20, value=9, step=1)

    # Parametri backtest
    with st.expander("Parametri Backtest", expanded=False):
        allow_short = st.checkbox("Abilita short (BUY=+1, SELL=-1, HOLD=0)", value=False)
        fee_bps = st.number_input("Costo per cambio posizione (bps)", min_value=0.0, max_value=50.0,
                                  value=2.0, step=0.5,
                                  help="1 bps = 0.01% per trade; applicato a ingressi/uscite/flip")

    # Pesi score fondamentale
    with st.expander("Pesi Score Fondamentale", expanded=False):
        w_val = st.slider("Valuation", 0, 100, 20, step=5)
        w_prof = st.slider("Profitability", 0, 100, 25, step=5)
        w_growth = st.slider("Growth", 0, 100, 30, step=5)
        w_sol = st.slider("Solidità", 0, 100, 15, step=5)
        w_div = st.slider("Dividendi", 0, 100, 10, step=5)
        w_sum = w_val + w_prof + w_growth + w_sol + w_div
        st.caption(f"Somma pesi: **{w_sum}** (verrà normalizzata automaticamente a 100)")

    run_analysis = st.button("🔎 Analizza")

# normalizza pesi (usati ovunque)
weights_raw = {
    "Valuation": w_val,
    "Profitability": w_prof,
    "Growth": w_growth,
    "Solidity": w_sol,
    "Dividends": w_div
}
weights_norm = {k: (v / (w_sum if w_sum > 0 else 1.0)) for k, v in weights_raw.items()}

with left:
    st.subheader("🗺️ Market Heatmap (Daily)")
    # Limita la heatmap ai primi N per market cap (quando disponibile)
    wl = st.session_state["watchlist"]
    df_heat = build_heatmap_df(wl)

    if not df_heat.empty:
        # Ordina per market cap e limita
        df_heat = df_heat.sort_values("MarketCap", ascending=False).head(heat_max)
        # Percentuale formattata con segno
        df_heat = df_heat.copy()
        df_heat["ChangePctStr"] = df_heat["Change"].apply(
            lambda x: f"+{x*100:.2f}%" if isinstance(x, (int, float)) and pd.notna(x) and x >= 0
            else (f"{x*100:.2f}%" if isinstance(x, (int, float)) and pd.notna(x) else "N/A")
        )

        fig_heat = px.treemap(
            df_heat,
            path=["Sector", "Ticker"],
            values="MarketCap",
            color="Change",
            color_continuous_scale=["#b00020", "#222222", "#00a86b"],
            color_continuous_midpoint=0,
            custom_data=["ChangePctStr"]
        )
        # Testo centrale: ticker (bold) + % sotto
        fig_heat.data[0].texttemplate = "<b>%{label}</b><br>%{customdata[0]}"
        fig_heat.data[0].textposition = "middle center"
        fig_heat.update_traces(textfont=dict(size=16))
        fig_heat.update_traces(marker=dict(line=dict(width=1, color="rgba(255,255,255,0.15)")))
        fig_heat.data[0].hovertemplate = (
            "<b>%{label}</b><br>"
            "Settore: %{parent}<br>"
            "Market Cap: %{value:,}<br>"
            "Performance: %{customdata[0]}<extra></extra>"
        )
        fig_heat.update_layout(
            template="plotly_dark",
            margin=dict(t=30, l=10, r=10, b=10),
            height=520,
            coloraxis_colorbar=dict(
                title="Δ Giornaliero",
                tickformat="+.1%",
                thickness=12,
                len=0.75
            )
        )
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.info("Nessun dato disponibile per la heatmap.")

st.markdown("---")

# =========================
# ANALISI DETTAGLIO
# =========================
if run_analysis or "last_signal" not in st.session_state:
    with st.spinner("Analisi in corso..."):
        data = get_history_normalized(ticker, period)

    if data is None or data.empty or "Close" not in data.columns:
        st.error("Errore nel recupero dati o colonna 'Close' assente.")
        if data is not None and not data.empty:
            st.write("Colonne disponibili:", list(data.columns))
        st.stop()

    if data.shape[0] < max(50, int(macd_slow) + 5):
        st.warning("Storico relativamente corto: indicatori a 50 periodi e MACD potrebbero essere meno affidabili.")

    # ===== TECNICO (parametric) =====
    close = data["Close"].astype(float)
    data["EMA_fast"] = ema(close, int(ema_fast))
    data["EMA_slow"] = ema(close, int(ema_slow))
    data["RSI14"] = rsi(close, 14)
    data["MACD"], data["MACD_signal"], data["MACD_hist"] = macd(close, int(macd_fast), int(macd_slow), int(macd_signal))
    data["KC_mid"], data["KC_upper"], data["KC_lower"] = keltner_channels(
        data, ema_period=int(ema_fast), atr_period=int(keltner_atr_period), atr_mult=float(keltner_mult)
    )
    data["ATR14"] = atr(data, int(atr_period))

    last = data.iloc[-1]
    last_price = float(last["Close"])
    atr14 = float(last["ATR14"]) if pd.notna(last["ATR14"]) else np.nan

    cond_trend_up   = last["EMA_fast"] > last["EMA_slow"]
    cond_trend_down = last["EMA_fast"] < last["EMA_slow"]
    cond_above_mid  = last_price > last["KC_mid"]
    cond_below_mid  = last_price < last["KC_mid"]
    cond_macd_bull  = last["MACD"] > last["MACD_signal"]
    cond_macd_bear  = last["MACD"] < last["MACD_signal"]
    cond_rsi_bull   = (last["RSI14"] > 50) if use_rsi else True
    cond_rsi_bear   = (last["RSI14"] < 50) if use_rsi else True

    if cond_trend_up and cond_above_mid and cond_macd_bull and cond_rsi_bull:
        tech_signal = "BUY"
    elif cond_trend_down and cond_below_mid and cond_macd_bear and cond_rsi_bear:
        tech_signal = "SELL"
    else:
        tech_signal = "HOLD"

    # ===== RISK MANAGEMENT (ATR) =====
    if pd.notna(atr14) and atr14 > 0:
        if tech_signal == "BUY":
            stop_loss = last_price - 1.5 * atr14
            take_profit_1 = last_price + 1.0 * atr14
            take_profit_2 = last_price + 2.0 * atr14
        elif tech_signal == "SELL":
            stop_loss = last_price + 1.5 * atr14
            take_profit_1 = last_price - 1.0 * atr14
            take_profit_2 = last_price - 2.0 * atr14
        else:
            stop_loss = take_profit_1 = take_profit_2 = np.nan
    else:
        stop_loss = take_profit_1 = take_profit_2 = np.nan

    # ===== Fondamentale (esteso con criteri + pesi) =====
    fund = get_fundamentals(ticker)
    snapshot = fund["snapshot"]
    series = fund["series"]
    criteria = fund["criteria"]

    # Score pesato (0..100) + band + motivazioni + contributi per categoria
    fscore100, fband, freasons, fcontrib = compute_weighted_score(criteria, weights_norm)

    # Classificazioni base per final_signal (cap_type / pe_status)
    t_obj = yf.Ticker(ticker)
    info_basic = safe_ticker_info(t_obj)
    market_cap = safe_get(info_basic, "marketCap")
    pe_ratio = safe_get(info_basic, "trailingPE")

    if market_cap:
        if market_cap > 10_000_000_000:
            cap_type = "Large Cap"
        elif market_cap > 2_000_000_000:
            cap_type = "Mid Cap"
        else:
            cap_type = "Small Cap"
    else:
        cap_type = "N/A"

    if isinstance(pe_ratio, (int, float)):
        if pe_ratio < 25:
            pe_status = "OK"
        elif pe_ratio < 40:
            pe_status = "MEDIUM"
        else:
            pe_status = "HIGH"
    else:
        pe_status = "N/A"

    # Usa lo score pesato per la validazione fondamentale
    fundamental_ok = fscore100 >= 60.0

    # ===== SEGNALE FINALE =====
    if tech_signal == "BUY" and fundamental_ok:
        final_signal = "BUY STRONG ⭐" if cap_type == "Large Cap" else "BUY STRONG"
    elif tech_signal == "BUY" and pe_status == "HIGH":
        final_signal = "BUY (overvalued)"
    elif tech_signal == "BUY":
        final_signal = "BUY (weak)"
    elif tech_signal == "SELL":
        final_signal = "SELL"
    else:
        final_signal = "HOLD"

    # =========================
    # SEGNALE STORICO (serie BUY/SELL/HOLD) per grafico e backtest
    # =========================
    df = data.copy()
    cond_up = (df["EMA_fast"] > df["EMA_slow"]) & (df["Close"] > df["KC_mid"]) \
              & (df["MACD"] > df["MACD_signal"]) & ((df["RSI14"] > 50) if use_rsi else True)
    cond_down = (df["EMA_fast"] < df["EMA_slow"]) & (df["Close"] < df["KC_mid"]) \
                & (df["MACD"] < df["MACD_signal"]) & ((df["RSI14"] < 50) if use_rsi else True)

    df["Signal"] = np.where(cond_up, "BUY", np.where(cond_down, "SELL", "HOLD"))
    df["Position"] = df["Signal"].replace({"BUY":1, "SELL":-1, "HOLD":0}).shift(1).fillna(0)
    df["CrossUp"] = (df["EMA_fast"] > df["EMA_slow"]) & (df["EMA_fast"].shift(1) <= df["EMA_slow"].shift(1))
    df["CrossDown"] = (df["EMA_fast"] < df["EMA_slow"]) & (df["EMA_fast"].shift(1) >= df["EMA_slow"].shift(1))

    # ===== TABS =====
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Tecnica", "📊 Fondamentale", "🎯 Consiglio", "📚 Backtest"])

    # TECNICA — grafico migliorato per leggere EMA_fast > EMA_slow + segnali
    with tab1:
        st.subheader("Analisi Tecnica (Trend + EMA spread)")
        st.write(
            f"Segnale tecnico: **{tech_signal}** — "
            f"EMA_fast({int(ema_fast)}): **{data['EMA_fast'].iloc[-1]:.2f}** | "
            f"EMA_slow({int(ema_slow)}): **{data['EMA_slow'].iloc[-1]:.2f}** | "
            f"Δ: **{(data['EMA_fast'].iloc[-1] - data['EMA_slow'].iloc[-1]):.2f}**"
        )

        fast = data["EMA_fast"]; slow = data["EMA_slow"]
        spread = fast - slow

        fig = go.Figure()
        # Prezzo
        fig.add_trace(go.Scatter(
            x=data.index, y=data["Close"], name="Prezzo", mode="lines",
            line=dict(color="#cfcfcf", width=1.5)
        ))
        # EMA slow (rossa, tratteggiata)
        fig.add_trace(go.Scatter(
            x=data.index, y=slow, name=f"EMA{int(ema_slow)} (slow)",
            mode="lines", line=dict(color="#FF6347", width=2, dash="dash")
        ))
        # EMA fast (oro)
        fig.add_trace(go.Scatter(
            x=data.index, y=fast, name=f"EMA{int(ema_fast)} (fast)",
            mode="lines", line=dict(color="#FFD700", width=2.5)
        ))

        # Fill verde dove fast>slow
        y_upper = np.where(fast > slow, fast, slow)
        y_lower = np.where(fast > slow, slow, fast)
        fig.add_trace(go.Scatter(x=data.index, y=y_upper, line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(
            x=data.index, y=y_lower, line=dict(width=0), showlegend=False, hoverinfo="skip",
            fill="tonexty", fillcolor="rgba(0, 170, 0, 0.10)", name="Zona fast>slow"
        ))
        # Fill rosso dove fast<slow
        y_upper_r = np.where(fast < slow, slow, np.nan)
        y_lower_r = np.where(fast < slow, fast, np.nan)
        fig.add_trace(go.Scatter(x=data.index, y=y_upper_r, line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(
            x=data.index, y=y_lower_r, line=dict(width=0), showlegend=False, hoverinfo="skip",
            fill="tonexty", fillcolor="rgba(200, 0, 0, 0.10)", name="Zona fast<slow"
        ))

        # Marker segnali operativi
        sig_buy_points = df[df["Signal"] == "BUY"]
        sig_sell_points = df[df["Signal"] == "SELL"]
        if not sig_buy_points.empty:
            fig.add_trace(go.Scatter(
                x=sig_buy_points.index, y=sig_buy_points["Close"],
                name="Signal BUY", mode="markers",
                marker=dict(color="#00c853", size=9, symbol="circle"),
                hovertemplate="BUY %{x|%Y-%m-%d}<br>Prezzo: %{y:.2f}<extra></extra>"
            ))
        if not sig_sell_points.empty:
            fig.add_trace(go.Scatter(
                x=sig_sell_points.index, y=sig_sell_points["Close"],
                name="Signal SELL", mode="markers",
                marker=dict(color="#d50000", size=9, symbol="x"),
                hovertemplate="SELL %{x|%Y-%m-%d}<br>Prezzo: %{y:.2f}<extra></extra>"
            ))

        fig.update_layout(
            template="plotly_dark", height=460,
            margin=dict(t=20, l=10, r=10, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
        )
        st.plotly_chart(fig, use_container_width=True)

        # Elenco segnali + download
        with st.expander("🧾 Elenco segnali (storico) + download"):
            sig_df = df.loc[(df["Signal"].shift(1) != df["Signal"]) | (df.index == df.index[0])].copy()
            sig_df = sig_df[["Signal", "Close", "EMA_fast", "EMA_slow", "MACD", "MACD_signal", "RSI14"]]
            sig_df = sig_df.rename(columns={
                "Close":"Prezzo", "EMA_fast":f"EMA{int(ema_fast)}", "EMA_slow":f"EMA{int(ema_slow)}"
            })
            st.dataframe(sig_df.tail(30))
            csv = sig_df.to_csv(index=True).encode("utf-8")
            st.download_button(
                label="⬇️ Scarica segnali (CSV)",
                data=csv,
                file_name=f"signals_{ticker}_{period}.csv",
                mime="text/csv"
            )

    # FONDAMENTALE (pesi custom)
    with tab2:
        st.subheader("Analisi Fondamentale (pesi personalizzati)")

        # Score complessivo
        color = "#3CB371" if fband == "Strong" else ("#F0AD4E" if fband == "Neutral" else "#D9534F")
        st.markdown(
            f"**Fundamental Score:** "
            f"<span style='color:{color}; font-size:20px'><b>{fscore100:.1f}/100 – {fband}</b></span>",
            unsafe_allow_html=True
        )
        with st.expander("Motivazioni (criteri rispettati)"):
            if freasons:
                for r in freasons:
                    st.write(f"• {r}")
            else:
                st.write("Nessun criterio soddisfatto (dati incompleti o metriche deboli).")

        # Snapshot KPI sintetico
        k1, k2, k3, k4, k5 = st.columns(5)
        with k1:
            st.caption("Valuation")
            st.metric("Forward P/E", f"{snapshot['Valuation']['Forward P/E']:.2f}" if isinstance(snapshot['Valuation']['Forward P/E'], (int,float)) else "N/A")
            st.metric("PEG", f"{snapshot['Valuation']['PEG']:.2f}" if isinstance(snapshot['Valuation']['PEG'], (int,float)) else "N/A")
        with k2:
            st.caption("Profitability")
            st.metric("ROE", fmt_pct(snapshot["Profitability"]["ROE"]))
            st.metric("Net Margin", fmt_pct(snapshot["Profitability"]["Net Margin"]))
        with k3:
            st.caption("Growth")
            st.metric("Rev CAGR", fmt_pct(snapshot["Growth"]["Revenue CAGR (annuale)"]))
            st.metric("EPS QoQ", fmt_pct(snapshot["Growth"]["EPS Growth QoQ"]))
        with k4:
            st.caption("Solidità")
            de = snapshot["Solidity"]["Debt/Equity"]
            st.metric("Debt/Equity", f"{de:.2f}" if isinstance(de, (int,float)) else "N/A")
            cr = snapshot["Solidity"]["Current Ratio"]
            st.metric("Current Ratio", f"{cr:.2f}" if isinstance(cr, (int,float)) else "N/A")
        with k5:
            st.caption("Dividendi")
            st.metric("Yield", fmt_pct(snapshot["Dividends"]["Dividend Yield"]))
            st.metric("Payout", fmt_pct(snapshot["Dividends"]["Payout Ratio"]))

        st.markdown("---")

        # Contributo per categoria e pesi
        contrib_df = pd.DataFrame({
            "Categoria": list(fcontrib.keys()),
            "Contributo (%)": [round(v, 2) for v in fcontrib.values()],
            "Peso (%)": [round(weights_norm[k]*100.0, 2) for k in fcontrib.keys()]
        })
        fig_contrib = go.Figure()
        fig_contrib.add_trace(go.Bar(
            x=contrib_df["Categoria"], y=contrib_df["Contributo (%)"],
            name="Contributo allo score", marker_color="#6aa84f"
        ))
        fig_contrib.add_trace(go.Scatter(
            x=contrib_df["Categoria"], y=contrib_df["Peso (%)"],
            name="Peso assegnato", mode="lines+markers", line=dict(color="#888", dash="dot")
        ))
        fig_contrib.update_layout(template="plotly_dark", height=320, margin=dict(t=20,l=10,r=10,b=10),
                                  yaxis_title="Percentuale")
        st.plotly_chart(fig_contrib, use_container_width=True)

        st.markdown("---")

        # Tabelle dettagliate
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Valuation**")
            v = snapshot["Valuation"]
            st.write({
                "Trailing P/E": v["Trailing P/E"],
                "Forward P/E": v["Forward P/E"],
                "PEG": v["PEG"],
                "P/S (TTM)": v["P/S (TTM)"],
                "P/B": v["P/B"],
                "EV/EBITDA": v["EV/EBITDA"],
                "EV/Revenue": v["EV/Revenue"]
            })

            st.markdown("**Profitability**")
            p = snapshot["Profitability"]
            st.write({
                "Gross Margin": fmt_pct(p["Gross Margin"]),
                "Operating Margin": fmt_pct(p["Operating Margin"]),
                "Net Margin": fmt_pct(p["Net Margin"]),
                "ROE": fmt_pct(p["ROE"]),
                "ROA": fmt_pct(p["ROA"])
            })

        with c2:
            st.markdown("**Growth**")
            g = snapshot["Growth"]
            st.write({
                "Revenue Growth (prov)": fmt_pct(g["Revenue Growth (prov)"]),
                "EPS Growth QoQ": fmt_pct(g["EPS Growth QoQ"]),
                "EPS Growth TTM": fmt_pct(g["EPS Growth TTM"]),
                "Revenue CAGR (annuale)": fmt_pct(g["Revenue CAGR (annuale)"])
            })

            st.markdown("**Solidità & Dividendi**")
            s = snapshot["Solidity"]; d = snapshot["Dividends"]
            st.write({
                "Debt/Equity": s["Debt/Equity"] if isinstance(s["Debt/Equity"], (int,float)) else "N/A",
                "Current Ratio": s["Current Ratio"] if isinstance(s["Current Ratio"], (int,float)) else "N/A",
                "Quick Ratio": s["Quick Ratio"] if isinstance(s["Quick Ratio"], (int,float)) else "N/A",
                "Total Debt": fmt_num(s["Total Debt"]),
                "Total Cash": fmt_num(s["Total Cash"]),
                "Dividend Yield": fmt_pct(d["Dividend Yield"]),
                "Payout Ratio": fmt_pct(d["Payout Ratio"]),
                "5Y Avg Yield": fmt_pct(d["5Y Avg Yield"])
            })

        st.markdown("---")

        # Grafici di trend Ricavi / Utili
        g1, g2 = st.columns(2)
        with g1:
            st.markdown("**Ricavi & Utile – Annuale**")
            try:
                df_earn = yf.Ticker(ticker).earnings
                if isinstance(df_earn, pd.DataFrame) and not df_earn.empty:
                    fig_a = go.Figure()
                    if "Revenue" in df_earn.columns:
                        fig_a.add_trace(go.Bar(x=df_earn.index.astype(str), y=df_earn["Revenue"], name="Revenue"))
                    if "Earnings" in df_earn.columns:
                        fig_a.add_trace(go.Bar(x=df_earn.index.astype(str), y=df_earn["Earnings"], name="Earnings"))
                    fig_a.update_layout(template="plotly_dark", barmode="group", height=320,
                                      margin=dict(t=10,l=10,r=10,b=10))
                    st.plotly_chart(fig_a, use_container_width=True)
                else:
                    st.info("Dati annuali non disponibili.")
            except Exception as e:
                st.warning(f"Dati annuali non disponibili: {e}")

        with g2:
            st.markdown("**Ricavi & Utile – Trimestrale**")
            try:
                qearn = yf.Ticker(ticker).quarterly_earnings
                if isinstance(qearn, pd.DataFrame) and not qearn.empty:
                    figq = go.Figure()
                    if "Revenue" in qearn.columns:
                        figq.add_trace(go.Scatter(x=qearn.index.astype(str), y=qearn["Revenue"],
                                                  name="Revenue", mode="lines+markers"))
                    if "Earnings" in qearn.columns:
                        figq.add_trace(go.Scatter(x=qearn.index.astype(str), y=qearn["Earnings"],
                                                  name="Earnings", mode="lines+markers"))
                    figq.update_layout(template="plotly_dark", height=320, margin=dict(t=10,l=10,r=10,b=10))
                    st.plotly_chart(figq, use_container_width=True)
                else:
                    st.info("Dati trimestrali non disponibili.")
            except Exception as e:
                st.warning(f"Dati trimestrali non disponibili: {e}")

    # CONSIGLIO
    with tab3:
        st.subheader("🎯 Consiglio Finale")
        if final_signal == "BUY STRONG ⭐":
            st.success("🔥 BUY STRONG ⭐")
        elif final_signal == "BUY STRONG":
            st.success("🔥 BUY STRONG")
        elif "BUY" in final_signal:
            st.info(final_signal)
        elif final_signal == "SELL":
            st.error("📉 SELL")
        else:
            st.warning("⚖️ HOLD")

        if final_signal in ("BUY STRONG ⭐", "BUY STRONG", "BUY (overvalued)", "BUY (weak)"):
            st.subheader("🎯 Livelli operativi (dinamici)")
            c1, c2 = st.columns(2)
            with c1:
                st.error("🛑 Stop Loss")
                st.write(f"{stop_loss:.2f}" if isinstance(stop_loss, (int,float)) and not np.isnan(stop_loss) else "N/A")
            with c2:
                st.success("🎯 Take Profit")
                st.write(f"Target 1: {take_profit_1:.2f}" if isinstance(take_profit_1, (int,float)) and not np.isnan(take_profit_1) else "N/A")
                st.write(f"Target 2: {take_profit_2:.2f}" if isinstance(take_profit_2, (int,float)) and not np.isnan(take_profit_2) else "N/A")
        st.caption("⚠️ Livelli indicativi, non consulenza finanziaria.")

    # BACKTEST
    with tab4:
        st.subheader("📚 Backtest")
        metrics, bt = backtest_from_signals(df["Close"], df["Signal"], allow_short=allow_short, fee_bps=float(fee_bps))

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("CAGR", f"{metrics['cagr']*100:,.2f}%" if pd.notna(metrics['cagr']) else "N/A")
        kpi2.metric("Max Drawdown", f"{metrics['max_drawdown']*100:,.2f}%" if pd.notna(metrics['max_drawdown']) else "N/A")
        kpi3.metric("Sharpe", f"{metrics['sharpe']:.2f}" if pd.notna(metrics['sharpe']) else "N/A")
        kpi4.metric("Win Rate", f"{metrics['win_rate']:.1f}%")

        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(x=bt.index, y=bt["Equity"], name="Strategia", mode="lines"))
        fig_eq.add_trace(go.Scatter(x=bt.index, y=(bt["Close"] / bt["Close"].iloc[0]), name="Buy & Hold",
                                    mode="lines", line=dict(color="#888", dash="dot")))
        fig_eq.update_layout(template="plotly_dark", height=420, margin=dict(t=20,l=10,r=10,b=10),
                             yaxis_title="Equity (base=1.0)")
        st.plotly_chart(fig_eq, use_container_width=True)

        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(x=bt.index, y=bt["Drawdown"], name="Drawdown",
                                    mode="lines", line=dict(color="#d9534f")))
        fig_dd.update_layout(template="plotly_dark", height=240, margin=dict(t=20,l=10,r=10,b=10),
                             yaxis_tickformat=".0%")
        st.plotly_chart(fig_dd, use_container_width=True)

        with st.expander("Dettagli numerici"):
            st.dataframe(bt.tail(10))
            st.json(metrics)
'''

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(app_code)

print('app.py written successfully with advanced watchlist.')