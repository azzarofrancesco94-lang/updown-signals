import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ===== CONFIG =====
st.set_page_config(page_title="UpDown Signals", layout="wide")

st.title("📊 UpDown Signals")
st.write("Analisi tecnica + fondamentale + gestione del rischio")

# ===== INPUT =====
col1, col2, col3= st.columns([1,1,2])

with col1:
    tickers = ["AAPL","MSFT","TSLA","AMZN","GOOGL","META","NVDA","NFLX","BBVA"]
with col2:
    ticker = st.selectbox("Seleziona un Titolo", tickers)
with col3:
    period = st.selectbox("Periodo", ["3mo","6mo","1y"])

# ===== ANALISI =====
if st.button("Analizza"):

    data = yf.download(ticker, period=period)
    data.columns = data.columns.droplevel(1)  # Flatten MultiIndex columns

    if data.empty:
        st.error("Errore nel recupero dati")
    else:

        # ===== TECNICO =====
        data["Return"] = data["Close"].pct_change()
        data["MA20"] = data["Close"].rolling(20).mean()
        std = data["Close"].rolling(20).std()

        data["Upper"] = data["MA20"] + 2 * std
        data["Lower"] = data["MA20"] - 2 * std

        last_price = data["Close"].iloc[-1]
        last_std = std.iloc[-1]
        ma20 = data["MA20"].iloc[-1]
        upper = data["Upper"].iloc[-1]
        lower = data["Lower"].iloc[-1]

        if last_price < lower:
            tech_signal = "BUY"
        elif last_price > upper:
            tech_signal = "SELL"
        else:
            tech_signal = "HOLD"

        # ===== FONDAMENTALE =====
        ticker_obj = yf.Ticker(ticker)
        info = ticker_obj.info

        eps_qoq = info.get("earningsQuarterlyGrowth", None)
        eps_ttm = info.get("earningsGrowth", None)
        market_cap = info.get("marketCap", None)
        pe_ratio = info.get("trailingPE", None)

        # Market Cap
        if market_cap:
            if market_cap > 10_000_000_000:
                cap_type = "Large Cap"
            elif market_cap > 2_000_000_000:
                cap_type = "Mid Cap"
            else:
                cap_type = "Small Cap"
        else:
            cap_type = "N/A"

        # P/E
        if pe_ratio:
            if pe_ratio < 25:
                pe_status = "OK"
            elif pe_ratio < 40:
                pe_status = "MEDIUM"
            else:
                pe_status = "HIGH"
        else:
            pe_status = "N/A"

        # Fondamentali OK
        fundamental_ok = False
        if eps_qoq and eps_ttm and pe_ratio:
            if eps_qoq > 0.10 and eps_ttm > 0.10 and pe_ratio < 25:
                fundamental_ok = True

        # ===== SEGNALE FINALE =====
        if tech_signal == "BUY" and fundamental_ok:
            if cap_type == "Large Cap":
                final_signal = "BUY STRONG ⭐"
            else:
                final_signal = "BUY STRONG"
        elif tech_signal == "BUY" and pe_status == "HIGH":
            final_signal = "BUY (overvalued)"
        elif tech_signal == "BUY":
            final_signal = "BUY (weak)"
        elif tech_signal == "SELL":
            final_signal = "SELL"
        else:
            final_signal = "HOLD"

        # ===== RISK MANAGEMENT DINAMICO =====
        stop_loss = last_price - (1.5 * last_std)
        take_profit_1 = ma20
        take_profit_2 = upper

        # ===== TABS =====
        tab1, tab2, tab3 = st.tabs(["📈 Tecnica", "📊 Fondamentale", "🎯 Consiglio"])

        # ===== TECNICA =====
        with tab1:
            st.subheader("Analisi Tecnica")
            st.write(f"Segnale: **{tech_signal}**")

            fig, ax = plt.subplots()
            ax.plot(data["Close"], label="Prezzo")
            ax.plot(data["MA20"], label="Media")
            ax.plot(data["Upper"], linestyle="--")
            ax.plot(data["Lower"], linestyle="--")
            ax.legend()

            st.pyplot(fig)

        # ===== FONDAMENTALE =====
        with tab2:
            st.subheader("Analisi Fondamentale")

            st.write(f"EPS QoQ: {eps_qoq:.2%}" if eps_qoq else "EPS QoQ: N/A")
            st.write(f"EPS TTM: {eps_ttm:.2%}" if eps_ttm else "EPS TTM: N/A")

            if market_cap:
                st.write(f"Market Cap: ${market_cap:,.0f} ({cap_type})")

            if pe_ratio:
                st.write(f"P/E: {pe_ratio:.2f} ({pe_status})")

        # ===== CONSIGLIO =====
        with tab3:
            st.subheader("🎯 Consiglio Finale")

            if final_signal == "BUY STRONG ⭐":
                st.success("🔥 BUY STRONG ⭐")
            elif final_signal == "BUY STRONG":
                st.success("🔥 BUY STRONG")
            elif "BUY" in final_signal:
                st.info(final_signal)
            elif final_signal == "SELL":
                st.error("📈 SELL")
            else:
                st.warning("⚖️ HOLD")

            # ===== LIVELLI DINAMICI =====
            if "BUY" in final_signal:

                st.subheader("🎯 Livelli operativi (dinamici)")

                col1, col2 = st.columns(2)

                with col1:
                    st.error("🛑 Stop Loss")
                    st.write(f"{stop_loss:.2f}")

                with col2:
                    st.success("🎯 Take Profit")
                    st.write(f"Target 1 (media): {take_profit_1:.2f}")
                    st.write(f"Target 2 (banda alta): {take_profit_2:.2f}")

                st.caption("⚠️ Livelli indicativi, non consulenza finanziaria")