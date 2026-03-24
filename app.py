# app.py
import re
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
    """Scarica lo storico yfinance e restituisce OHLCV normalizzato."""
    raw = yf.download(ticker, period=period, progress=False, auto_adjust=False, group_by="column")
    if raw is None or raw.empty:
        return pd.DataFrame()
    return normalize_ohlcv(raw)

# =========================
# LAYOUT: HEATMAP + CONTROLLI AFFIANCATI
# =========================
left, right = st.columns([3, 1], gap="large")

with left:
    st.subheader("🗺️ Market Heatmap (Daily)")
    df_heat = build_heatmap_df(HEATMAP_TICKERS)

    if not df_heat.empty:
        fig_heat = px.treemap(
            df_heat,
            path=["Sector","Ticker"],
            values="MarketCap",
            color="Change",
            color_continuous_scale=["red","black","green"],
            color_continuous_midpoint=0
        )
        fig_heat.update_layout(template="plotly_dark", margin=dict(t=30, l=10, r=10, b=10), height=520)
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.info("Nessun dato disponibile per la heatmap.")

with right:
    st.subheader("⚙️ Impostazioni")
    ticker = st.selectbox("Seleziona un Titolo", HEATMAP_TICKERS, index=0)
    period = st.selectbox("Periodo", ["3mo","6mo","1y"], index=0)
    run_analysis = st.button("🔎 Analizza")

st.markdown("---")

# =========================
# ANALISI DETTAGLIO
# =========================
if run_analysis:
    with st.spinner("Analisi in corso..."):
        data = get_history_normalized(ticker, period)

    if data is None or data.empty or "Close" not in data.columns:
        st.error("Errore nel recupero dati o colonna 'Close' assente.")
        if data is not None and not data.empty:
            st.write("Colonne disponibili:", list(data.columns))
    else:
        # Guardrail: servono almeno 50 barre per avere indicatori stabili (EMA50, MACD)
        if data.shape[0] < 50:
            st.warning("Storico relativamente corto: indicatori a 50 periodi e MACD potrebbero essere meno affidabili.")

        # ===== TECNICO (EMA/Keltner/MACD/RSI + ATR) =====
        # Richiede: Open, High, Low, Close
        close = data["Close"].astype(float)

        # Trend & momentum
        data["EMA20"] = ema(close, 20)
        data["EMA50"] = ema(close, 50)
        data["RSI14"] = rsi(close, 14)
        data["MACD"], data["MACD_signal"], data["MACD_hist"] = macd(close, 12, 26, 9)

        # Keltner (volatilità “canalizzata”)
        data["KC_mid"], data["KC_upper"], data["KC_lower"] = keltner_channels(data, ema_period=20, atr_period=10, atr_mult=2.0)

        # ATR per risk management
        data["ATR14"] = atr(data, 14)

        last = data.iloc[-1]
        last_price = float(last["Close"])
        atr14 = float(last["ATR14"]) if pd.notna(last["ATR14"]) else np.nan

        # Regole segnale:
        # BUY  se trend UP (EMA20>EMA50), prezzo sopra KC_mid, MACD>signal e RSI>50
        # SELL se trend DOWN (EMA20<EMA50), prezzo sotto KC_mid, MACD<signal e RSI<50
        cond_trend_up   = last["EMA20"] > last["EMA50"]
        cond_trend_down = last["EMA20"] < last["EMA50"]
        cond_above_mid  = last_price > last["KC_mid"]
        cond_below_mid  = last_price < last["KC_mid"]
        cond_macd_bull  = last["MACD"] > last["MACD_signal"]
        cond_macd_bear  = last["MACD"] < last["MACD_signal"]
        cond_rsi_bull   = last["RSI14"] > 50
        cond_rsi_bear   = last["RSI14"] < 50

        if cond_trend_up and cond_above_mid and cond_macd_bull and cond_rsi_bull:
            tech_signal = "BUY"
        elif cond_trend_down and cond_below_mid and cond_macd_bear and cond_rsi_bear:
            tech_signal = "SELL"
        else:
            tech_signal = "HOLD"

        # ===== FONDAMENTALE =====
        t_obj = yf.Ticker(ticker)
        info = safe_ticker_info(t_obj)

        eps_qoq = info.get("earningsQuarterlyGrowth", None)
        eps_ttm = info.get("earningsGrowth", None)
        market_cap = info.get("marketCap", None)
        pe_ratio = info.get("trailingPE", None)

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

        fundamental_ok = False
        if isinstance(eps_qoq, (int, float)) and isinstance(eps_ttm, (int, float)) and isinstance(pe_ratio, (int, float)):
            if eps_qoq > 0.10 and eps_ttm > 0.10 and pe_ratio < 25:
                fundamental_ok = True

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

        # ===== TABS =====
        tab1, tab2, tab3 = st.tabs(["📈 Tecnica", "📊 Fondamentale", "🎯 Consiglio"])

        # TECNICA
        with tab1:
            st.subheader("Analisi Tecnica (EMA20/EMA50 + Keltner + MACD/RSI)")
            st.write(f"Segnale: **{tech_signal}**")

            # Prezzo + EMA + Keltner
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data.index, y=data["Close"], name="Prezzo", mode="lines"))
            if data["EMA20"].notna().any():
                fig.add_trace(go.Scatter(x=data.index, y=data["EMA20"], name="EMA20", mode="lines"))
            if data["EMA50"].notna().any():
                fig.add_trace(go.Scatter(x=data.index, y=data["EMA50"], name="EMA50", mode="lines"))
            if data["KC_upper"].notna().any():
                fig.add_trace(go.Scatter(x=data.index, y=data["KC_upper"], name="Keltner Alta", mode="lines",
                                         line=dict(dash="dash", color="#888")))
            if data["KC_lower"].notna().any():
                fig.add_trace(go.Scatter(x=data.index, y=data["KC_lower"], name="Keltner Bassa", mode="lines",
                                         line=dict(dash="dash", color="#888")))
            if data["KC_mid"].notna().any():
                fig.add_trace(go.Scatter(x=data.index, y=data["KC_mid"], name="Keltner Mid", mode="lines",
                                         line=dict(color="#aaa")))
            fig.update_layout(template="plotly_dark", height=420, margin=dict(t=20,l=10,r=10,b=10))
            st.plotly_chart(fig, use_container_width=True)

            # Sotto-pannello: MACD e RSI
            c1, c2 = st.columns(2)
            with c1:
                fig_macd = go.Figure()
                fig_macd.add_trace(go.Scatter(x=data.index, y=data["MACD"], name="MACD", mode="lines"))
                fig_macd.add_trace(go.Scatter(x=data.index, y=data["MACD_signal"], name="Signal", mode="lines"))
                # Istogramma
                fig_macd.add_trace(go.Bar(x=data.index, y=data["MACD_hist"], name="Hist", marker_color="#6aa84f"))
                fig_macd.update_layout(template="plotly_dark", height=260, margin=dict(t=20,l=10,r=10,b=10))
                st.plotly_chart(fig_macd, use_container_width=True)
            with c2:
                fig_rsi = go.Figure()
                fig_rsi.add_trace(go.Scatter(x=data.index, y=data["RSI14"], name="RSI(14)", mode="lines"))
                fig_rsi.add_hline(y=70, line=dict(color="#a64d79", dash="dash"))
                fig_rsi.add_hline(y=30, line=dict(color="#3d85c6", dash="dash"))
                fig_rsi.update_layout(template="plotly_dark", height=260, margin=dict(t=20,l=10,r=10,b=10),
                                      yaxis=dict(range=[0,100]))
                st.plotly_chart(fig_rsi, use_container_width=True)

        # FONDAMENTALE
        with tab2:
            st.subheader("Analisi Fondamentale")
            def fmt_pct(x): return f"{x:.2%}" if isinstance(x, (int,float)) and pd.notna(x) else "N/A"
            def fmt_num(x): return f"${x:,.0f}" if isinstance(x, (int,float)) and pd.notna(x) else "N/A"
            def fmt_pe(x):  return f"{x:.2f}" if isinstance(x, (int,float)) and pd.notna(x) else "N/A"

            st.write(f"EPS QoQ: {fmt_pct(eps_qoq)}")
            st.write(f"EPS TTM: {fmt_pct(eps_ttm)}")
            st.write(f"Market Cap: {fmt_num(market_cap)} ({cap_type})")
            st.write(f"P/E: {fmt_pe(pe_ratio)} ({pe_status})")

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