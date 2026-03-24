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
# (1) HEATMAP PRIMA DEI MENU
# =========================
st.subheader("🗺️ Market Heatmap (Daily)")

# Lista tickers “global” per la heatmap
HEATMAP_TICKERS = ["AAPL","MSFT","TSLA","AMZN","GOOGL","META","NVDA","NFLX","BBVA"]

heatmap_data = []
for t in HEATMAP_TICKERS:
    try:
        stock = yf.Ticker(t)
        info = safe_ticker_info(stock)

        price = info.get("currentPrice", None)
        prev = info.get("previousClose", None)
        market_cap = info.get("marketCap", 1)
        sector = info.get("sector", "Other")

        if isinstance(price, (int, float)) and isinstance(prev, (int, float)) and prev not in (0, None):
            change = (price - prev) / prev
            heatmap_data.append({
                "Sector": sector or "Other",
                "Ticker": t,
                "Change": change,
                "MarketCap": float(market_cap) if market_cap else 1.0
            })
    except Exception as e:
        st.warning(f"Heatmap: impossibile processare {t}: {e}")

df_heat = pd.DataFrame(heatmap_data)

if not df_heat.empty:
    fig_heat = px.treemap(
        df_heat,
        path=["Sector","Ticker"],
        values="MarketCap",
        color="Change",
        color_continuous_scale=["red","black","green"],
        color_continuous_midpoint=0
    )
    fig_heat.update_layout(template="plotly_dark", margin=dict(t=30, l=10, r=10, b=10))
    st.plotly_chart(fig_heat, use_container_width=True)
else:
    st.info("Nessun dato disponibile per la heatmap.")

st.markdown("---")

# =========================
# (2) MENU A TENDINA DOPO LA HEATMAP
# =========================
col1, col2, col3 = st.columns([1,1,2])
with col1:
    tickers = HEATMAP_TICKERS  # riusa la stessa lista
with col2:
    ticker = st.selectbox("Seleziona un Titolo", tickers, index=0)
with col3:
    period = st.selectbox("Periodo", ["3mo","6mo","1y"], index=0)

# =========================
# ANALISI DETTAGLIO
# =========================
if st.button("Analizza"):
    data_raw = yf.download(ticker, period=period, progress=False, auto_adjust=False, group_by="column")

    if data_raw is None or data_raw.empty:
        st.error("Errore nel recupero dati")
    else:
        data = normalize_ohlcv(data_raw)

        if data is None or data.empty or "Close" not in data.columns:
            st.error("La colonna 'Close' non è presente nei dati scaricati.")
            st.write("Colonne originali:", list(flatten_columns(data_raw).columns))
            st.stop()

        # Guardrail: servono almeno 20 barre
        if data.shape[0] < 20:
            st.warning("Storico troppo corto per calcolare indicatori a 20 periodi con affidabilità.")

        # ===== TECNICO =====
        close = data["Close"].astype(float)
        ma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        upper = ma20 + 2 * std20
        lower = ma20 - 2 * std20

        last_price = float(close.iloc[-1])
        last_std = float(std20.iloc[-1]) if pd.notna(std20.iloc[-1]) else np.nan
        last_ma20 = float(ma20.iloc[-1]) if pd.notna(ma20.iloc[-1]) else np.nan
        last_upper = float(upper.iloc[-1]) if pd.notna(upper.iloc[-1]) else np.nan
        last_lower = float(lower.iloc[-1]) if pd.notna(lower.iloc[-1]) else np.nan

        if pd.notna(last_lower) and last_price < last_lower:
            tech_signal = "BUY"
        elif pd.notna(last_upper) and last_price > last_upper:
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

        stop_loss = last_price - (1.5 * last_std) if pd.notna(last_std) and last_std > 0 else np.nan
        take_profit_1 = last_ma20
        take_profit_2 = last_upper

        tab1, tab2, tab3 = st.tabs(["📈 Tecnica", "📊 Fondamentale", "🎯 Consiglio"])

        with tab1:
            st.subheader("Analisi Tecnica")
            st.write(f"Segnale: **{tech_signal}**")
            fig, ax = plt.subplots()
            ax.plot(close.index, close.values, label="Prezzo")
            if ma20.notna().any():
                ax.plot(ma20.index, ma20.values, label="MA20")
            if upper.notna().any():
                ax.plot(upper.index, upper.values, linestyle="--", label="Banda alta")
            if lower.notna().any():
                ax.plot(lower.index, lower.values, linestyle="--", label="Banda bassa")
            ax.legend()
            st.pyplot(fig)

        with tab2:
            st.subheader("Analisi Fondamentale")
            st.write(f"EPS QoQ: {eps_qoq:.2%}" if isinstance(eps_qoq, (int,float)) else "EPS QoQ: N/A")
            st.write(f"EPS TTM: {eps_ttm:.2%}" if isinstance(eps_ttm, (int,float)) else "EPS TTM: N/A")
            if market_cap:
                st.write(f"Market Cap: ${market_cap:,.0f} ({cap_type})")
            st.write(f"P/E: {pe_ratio:.2f} ({pe_status})" if isinstance(pe_ratio, (int,float)) else f"P/E: N/A ({pe_status})")

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

            if "BUY" in final_signal:
                st.subheader("🎯 Livelli operativi (dinamici)")
                c1, c2 = st.columns(2)
                with c1:
                    st.error("🛑 Stop Loss")
                    st.write(f"{stop_loss:.2f}" if isinstance(stop_loss, (int,float)) and not np.isnan(stop_loss) else "N/A")
                with c2:
                    st.success("🎯 Take Profit")
                    st.write(f"Target 1 (media): {take_profit_1:.2f}" if isinstance(take_profit_1, (int,float)) and not np.isnan(take_profit_1) else "N/A")
                    st.write(f"Target 2 (banda alta): {take_profit_2:.2f}" if isinstance(take_profit_2, (int,float)) and not np.isnan(take_profit_2) else "N/A")

            st.caption("⚠️ Livelli indicativi, non consulenza finanziaria.")