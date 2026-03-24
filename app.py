import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import timedelta
import plotly.express as px
import plotly.graph_objects as go

# ============ CONFIG ============
st.set_page_config(page_title="UpDown Signals", layout="wide")
st.title("📊 UpDown Signals")
st.write("Il tuo assistente per segnali di trading basati su analisi tecnica e fondamentale")
st.caption("⚠️ Educational only – non è consulenza finanziaria.")

# ============ INPUT ============
col1, col2, col3 = st.columns([1,1,2])
with col1:
    TICKERS = ["AAPL","MSFT","TSLA","AMZN","GOOGL","META","NVDA","NFLX","BBVA"]
with col2:
    ticker = st.selectbox("Seleziona un Titolo", TICKERS, index=0)
with col3:
    period = st.selectbox("Periodo", ["3mo","6mo","1y"], index=0)

# ============ UTILS ============
@st.cache_data(ttl=300)
def fetch_prices_for_heatmap(tickers):
    """Scarica prezzi ultimi 2 giorni per tutti i tickers (unica call)."""
    # 2d per avere prev close + ultimo close
    df = yf.download(tickers, period="2d", group_by="ticker", auto_adjust=False, progress=False)
    return df

@st.cache_data(ttl=300)
def fetch_fast_info(ticker):
    t = yf.Ticker(ticker)
    # fast_info è più snello e stabile per alcuni campi
    fi = getattr(t, "fast_info", {}) or {}
    # fallback a info solo per campi mancanti
    info = {}
    try:
        info = t.info or {}
    except Exception:
        info = {}
    return fi, info

def get_close_and_prev_from_grouped(df, t):
    """Estrae last close e prev close da df group_by='ticker' robustamente."""
    if isinstance(df.columns, pd.MultiIndex):
        # Struttura: livello1=ticker, livello2=campo (Open/High/Low/Close/...)
        if t in df.columns.get_level_values(0):
            sub = df[t]
            if "Close" in sub.columns:
                closes = sub["Close"].dropna()
                if len(closes) >= 2:
                    return closes.iloc[-1], closes.iloc[-2]
                elif len(closes) == 1:
                    return closes.iloc[-1], np.nan
    else:
        # Caso singolo ticker (non usato qui, ma utile)
        closes = df["Close"].dropna()
        if len(closes) >= 2:
            return closes.iloc[-1], closes.iloc[-2]
        elif len(closes) == 1:
            return closes.iloc[-1], np.nan
    return np.nan, np.nan

def compute_atr(df, period=14):
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return atr

# ============ HEATMAP ============
st.subheader("🗺️ Market Heatmap (Daily)")

heatmap_rows = []
with st.spinner("Caricamento dati mercato..."):
    try:
        df_all = fetch_prices_for_heatmap(TICKERS)
        for t in TICKERS:
            last_c, prev_c = get_close_and_prev_from_grouped(df_all, t)
            if pd.notna(last_c) and pd.notna(prev_c) and prev_c != 0:
                change = (last_c - prev_c) / prev_c
            else:
                change = np.nan

            # MarketCap: usa fast_info; se mancante, fallback a info
            fi, info = fetch_fast_info(t)
            mcap = fi.get("market_cap")
            if mcap is None:
                mcap = info.get("marketCap", 1)
            sector = info.get("sector") or "Other"

            if pd.notna(change):
                heatmap_rows.append({
                    "Sector": sector,
                    "Ticker": t,
                    "Change": change,
                    "MarketCap": float(mcap) if mcap else 1.0
                })
    except Exception as e:
        st.warning(f"Impossibile costruire heatmap: {e}")

df_heat = pd.DataFrame(heatmap_rows)

if not df_heat.empty:
    fig_heat = px.treemap(
        df_heat,
        path=["Sector","Ticker"],
        values="MarketCap",
        color="Change",
        color_continuous_scale=["red","black","green"],
        color_continuous_midpoint=0
    )
    fig_heat.update_layout(template="plotly_dark", margin=dict(t=30,l=10,r=10,b=10))
    st.plotly_chart(fig_heat, use_container_width=True)
else:
    st.info("Nessun dato disponibile per la heatmap.")

st.markdown("---")

# ============ ANALISI ============
if st.button("Analizza"):
    with st.spinner("Analisi in corso..."):
        # Scarico storico singolo ticker per periodo richiesto (+ buffer)
        data = yf.download(ticker, period=period, progress=False, auto_adjust=False)
        if data is None or data.empty:
            st.error("Errore nel recupero dati")
        else:
            # Normalizza colonne (evita droplevel che può crashare)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = ["_".join(col).strip() if isinstance(col, tuple) else col
                                for col in data.columns.values]
                # Prova a ricostruire le canoniche
                rename_map = {}
                for c in data.columns:
                    if c.endswith("_Close"): rename_map[c] = "Close"
                    if c.endswith("_Open"):  rename_map[c] = "Open"
                    if c.endswith("_High"):  rename_map[c] = "High"
                    if c.endswith("_Low"):   rename_map[c] = "Low"
                    if c.endswith("_Volume"):rename_map[c] = "Volume"
                data = data.rename(columns=rename_map)

            # Guardrail per lunghezza
            if data.shape[0] < 25:
                st.warning("Storico troppo corto per calcolare indicatori a 20 periodi con affidabilità.")
            
            # ===== TECNICO (Bollinger + ATR) =====
            data = data.sort_index()
            data["MA20"] = data["Close"].rolling(20).mean()
            std20 = data["Close"].rolling(20).std()
            data["Upper"] = data["MA20"] + 2 * std20
            data["Lower"] = data["MA20"] - 2 * std20

            data["ATR14"] = compute_atr(data, 14)

            last_price = float(data["Close"].iloc[-1])
            ma20 = float(data["MA20"].iloc[-1]) if pd.notna(data["MA20"].iloc[-1]) else np.nan
            upper = float(data["Upper"].iloc[-1]) if pd.notna(data["Upper"].iloc[-1]) else np.nan
            lower = float(data["Lower"].iloc[-1]) if pd.notna(data["Lower"].iloc[-1]) else np.nan
            atr14 = float(data["ATR14"].iloc[-1]) if pd.notna(data["ATR14"].iloc[-1]) else np.nan

            if pd.notna(lower) and last_price < lower:
                tech_signal = "BUY"
            elif pd.notna(upper) and last_price > upper:
                tech_signal = "SELL"
            else:
                tech_signal = "HOLD"

            # ===== FONDAMENTALE (robust fallback) =====
            fi, info = fetch_fast_info(ticker)
            pe = info.get("trailingPE") or info.get("forwardPE") or fi.get("trailing_pe") or None
            eps_qoq = info.get("earningsQuarterlyGrowth")
            eps_ttm = info.get("earningsGrowth")  # può essere None/sporco
            market_cap = (fi.get("market_cap")
                          or info.get("marketCap"))
            # classificazione cap
            cap_type = "N/A"
            if market_cap:
                if market_cap > 10_000_000_000: cap_type = "Large Cap"
                elif market_cap > 2_000_000_000: cap_type = "Mid Cap"
                else: cap_type = "Small Cap"

            # P/E status
            if pe is None:
                pe_status = "N/A"
            else:
                pe_status = "OK" if pe < 25 else ("MEDIUM" if pe < 40 else "HIGH")

            # scoring fondamentale leggermente più morbido
            fundamental_score = 0
            if isinstance(eps_qoq, (int, float)) and eps_qoq > 0.05: fundamental_score += 1
            if isinstance(eps_ttm, (int, float)) and eps_ttm > 0.05: fundamental_score += 1
            if isinstance(pe, (int, float)) and pe < 25: fundamental_score += 1

            fundamental_ok = fundamental_score >= 2  # soglia morbida

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

            # ===== RISK MANAGEMENT (ATR-based) =====
            if pd.notna(atr14) and atr14 > 0:
                stop_loss = last_price - 1.5 * atr14
                tp1 = last_price + 1.0 * atr14
                tp2 = last_price + 2.0 * atr14
            else:
                # fallback su std20
                stop_loss = last_price - (1.5 * std20.iloc[-1] if pd.notna(std20.iloc[-1]) else 0.0)
                tp1 = ma20 if pd.notna(ma20) else np.nan
                tp2 = upper if pd.notna(upper) else np.nan

            # ===== TABS =====
            tab1, tab2, tab3 = st.tabs(["📈 Tecnica", "📊 Fondamentale", "🎯 Consiglio"])

            with tab1:
                st.subheader("Analisi Tecnica")
                st.write(f"Segnale: **{tech_signal}**")

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=data.index, y=data["Close"], name="Prezzo", mode="lines"))
                if data["MA20"].notna().any():
                    fig.add_trace(go.Scatter(x=data.index, y=data["MA20"], name="MA20", mode="lines"))
                if data["Upper"].notna().any():
                    fig.add_trace(go.Scatter(x=data.index, y=data["Upper"], name="Banda Alta", mode="lines", line=dict(dash="dash")))
                if data["Lower"].notna().any():
                    fig.add_trace(go.Scatter(x=data.index, y=data["Lower"], name="Banda Bassa", mode="lines", line=dict(dash="dash")))
                fig.update_layout(template="plotly_dark", height=420, margin=dict(t=20,l=10,r=10,b=10))
                st.plotly_chart(fig, use_container_width=True)

            with tab2:
                st.subheader("Analisi Fondamentale")
                def fmt_pct(x): 
                    return f"{x:.2%}" if isinstance(x, (int,float)) and pd.notna(x) else "N/A"
                def fmt_num(x): 
                    return f"${x:,.0f}" if isinstance(x, (int,float)) and pd.notna(x) else "N/A"
                def fmt_pe(x): 
                    return f"{x:.2f}" if isinstance(x, (int,float)) and pd.notna(x) else "N/A"

                st.write(f"EPS QoQ: {fmt_pct(eps_qoq)}")
                st.write(f"EPS TTM: {fmt_pct(eps_ttm)}")
                st.write(f"Market Cap: {fmt_num(market_cap)} ({cap_type})")
                st.write(f"P/E: {fmt_pe(pe)} ({pe_status})")

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

                if "BUY" in final_signal or final_signal == "HOLD":
                    st.subheader("🎯 Livelli operativi (dinamici)")
                    c1, c2 = st.columns(2)
                    with c1:
                        st.error("🛑 Stop Loss")
                        st.write(f"{stop_loss:.2f}" if pd.notna(stop_loss) else "N/A")
                    with c2:
                        st.success("🎯 Take Profit")
                        st.write(f"Target 1: {tp1:.2f}" if pd.notna(tp1) else "N/A")
                        st.write(f"Target 2: {tp2:.2f}" if pd.notna(tp2) else "N/A")
                st.caption("⚠️ Livelli indicativi, non consulenza finanziaria.")