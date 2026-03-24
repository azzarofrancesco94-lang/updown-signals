# app.py
import re
import os
import ssl
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

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

# ---------- Backtest ----------
def backtest_from_signals(close: pd.Series,
                          signal_series: pd.Series,
                          allow_short: bool = False,
                          fee_bps: float = 0.0):
    """
    Esegue un backtest semplice daily:
    - positions: 1 per BUY, 0 HOLD (o -1 per SELL se allow_short)
    - execution: a mercato successivo (shift delle posizioni)
    - costi: fee_bps applicato al cambio posizione (entry/exit/flip)
    Ritorna: dict con metrics + DataFrame con equity e drawdown.
    """
    # Ritorni logici
    ret = close.pct_change().fillna(0.0)

    if allow_short:
        pos = np.where(signal_series == "BUY", 1,
                       np.where(signal_series == "SELL", -1, 0))
    else:
        pos = np.where(signal_series == "BUY", 1, 0)

    pos = pd.Series(pos, index=close.index).shift(1).fillna(0.0)  # no look-ahead

    # Costi transazione su cambi posizione
    pos_change = pos.diff().abs().fillna(pos.abs())
    fee = (fee_bps / 10_000.0) * pos_change

    strat_ret = (pos * ret) - fee
    equity = (1 + strat_ret).cumprod()
    high_watermark = equity.cummax()
    drawdown = (equity / high_watermark) - 1.0
    max_dd = drawdown.min()

    # Metriche annualizzate (assume ~252 giorni borsa/anno)
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

# ---------- Notifiche email ----------
def send_email(subject: str, body_html: str, recipients: list[str]) -> tuple[bool, str]:
    """Invia email via SMTP usando st.secrets[email]."""
    try:
        cfg = st.secrets.get("email", {})
        host = cfg.get("smtp_host")
        port = int(cfg.get("smtp_port", 587))
        user = cfg.get("smtp_user")
        pwd  = cfg.get("smtp_password")
        sender = cfg.get("sender", user)

        if not (host and port and user and pwd and sender):
            return False, "Config email mancante in st.secrets[email]."

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = sender
        msg["To"] = ", ".join(recipients)
        msg.attach(MIMEText(body_html, "html"))

        context = ssl.create_default_context()
        with smtplib.SMTP(host, port) as server:
            server.starttls(context=context)
            server.login(user, pwd)
            server.sendmail(sender, recipients, msg.as_string())

        return True, "Email inviata."
    except Exception as e:
        return False, f"Errore invio email: {e}"

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
        fee_bps = st.number_input("Costo per cambio posizione (bps)", min_value=0.0, max_value=50.0, value=2.0, step=0.5,
                                  help="1 bps = 0.01% per trade; applicato a ingressi/uscite/flip")

    # Notifiche email
    with st.expander("Notifiche Email", expanded=False):
        enable_email = st.checkbox("Invia email quando il segnale cambia", value=False)
        default_rcpt = st.secrets.get("email", {}).get("default_recipients", "")
        recipients_text = st.text_input("Destinatari (separati da virgola)", value=default_rcpt)
        test_email = st.button("Invia email di test")

    run_analysis = st.button("🔎 Analizza")

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

    # Guardrail min barre per indicatori a 50 periodi e MACD
    if data.shape[0] < max(ema_slow, macd_slow) + 5:
        st.warning("Storico relativamente corto: indicatori potrebbero essere meno affidabili.")

    # ===== TECNICO (parametric) =====
    close = data["Close"].astype(float)
    data["EMA_fast"] = ema(close, ema_fast)
    data["EMA_slow"] = ema(close, ema_slow)
    data["RSI14"] = rsi(close, 14)
    data["MACD"], data["MACD_signal"], data["MACD_hist"] = macd(close, macd_fast, macd_slow, macd_signal)
    data["KC_mid"], data["KC_upper"], data["KC_lower"] = keltner_channels(
        data, ema_period=ema_fast, atr_period=keltner_atr_period, atr_mult=keltner_mult
    )
    data["ATR14"] = atr(data, atr_period)

    last = data.iloc[-1]
    last_price = float(last["Close"])
    atr14 = float(last["ATR14"]) if pd.notna(last["ATR14"]) else np.nan

    # Regole segnale (parametriche)
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

    # ===== Segnale fondamentale (semplice come in precedenza) =====
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

    # Segnale finale
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

    # ======= NOTIFICHE EMAIL =======
    if enable_email:
        rcpts = [x.strip() for x in recipients_text.split(",") if x.strip()]
        last_key = f"last_signal_{ticker}"
        prev_sig = st.session_state.get(last_key)

        if test_email:
            ok, msg = send_email(
                subject=f"[TEST] UpDown Signals - {ticker}",
                body_html=f"<h3>Test email</h3><p>App attiva. Ultimo segnale calcolato: <b>{final_signal}</b></p>",
                recipients=rcpts
            )
            st.info(msg if ok else f"❌ {msg}")

        if prev_sig != final_signal and rcpts:
            html = f"""
            <h3>Segnale aggiornato per {ticker}</h3>
            <p><b>Nuovo segnale:</b> {final_signal}</p>
            <ul>
                <li><b>Tecnico:</b> {tech_signal}</li>
                <li><b>Prezzo:</b> {last_price:.2f}</li>
                <li><b>SL:</b> {stop_loss if pd.notna(stop_loss) else 'N/A'}</li>
                <li><b>TP1:</b> {take_profit_1 if pd.notna(take_profit_1) else 'N/A'}</li>
                <li><b>TP2:</b> {take_profit_2 if pd.notna(take_profit_2) else 'N/A'}</li>
            </ul>
            <p><b>Parametri:</b> EMA_fast={ema_fast}, EMA_slow={ema_slow}, ATR={atr_period}, Keltner_ATR={keltner_atr_period}, Mult={keltner_mult}, MACD=({macd_fast},{macd_slow},{macd_signal}), RSI_filter={use_rsi}</p>
            """
            ok, msg = send_email(subject=f"UpDown Signals - {ticker}: {final_signal}",
                                 body_html=html,
                                 recipients=rcpts)
            st.info(msg if ok else f"❌ {msg}")

        st.session_state[last_key] = final_signal

    # ===== TABS =====
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Tecnica", "📊 Fondamentale", "🎯 Consiglio", "📚 Backtest"])

    # TECNICA
    with tab1:
        st.subheader("Analisi Tecnica (parametric)")
        st.write(f"Segnale tecnico: **{tech_signal}**  |  Segnale finale: **{final_signal}**")

        # Prezzo + EMA + Keltner
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data["Close"], name="Prezzo", mode="lines"))
        if data["EMA_fast"].notna().any():
            fig.add_trace(go.Scatter(x=data.index, y=data["EMA_fast"], name=f"EMA{ema_fast}", mode="lines"))
        if data["EMA_slow"].notna().any():
            fig.add_trace(go.Scatter(x=data.index, y=data["EMA_slow"], name=f"EMA{ema_slow}", mode="lines"))
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

    # BACKTEST
    with tab4:
        st.subheader("📚 Backtest")
        # Costruisci serie segnali storici con regole parametriche
        df = data.copy()
        # condizioni bool su tutta la serie
        cond_up = (df["EMA_fast"] > df["EMA_slow"]) & (df["Close"] > df["KC_mid"])
        cond_down = (df["EMA_fast"] < df["EMA_slow"]) & (df["Close"] < df["KC_mid"])
        cond_macd_bull_s = (df["MACD"] > df["MACD_signal"])
        cond_macd_bear_s = (df["MACD"] < df["MACD_signal"])
        if use_rsi:
            cond_rsi_bull_s = (df["RSI14"] > 50)
            cond_rsi_bear_s = (df["RSI14"] < 50)
        else:
            cond_rsi_bull_s = pd.Series(True, index=df.index)
            cond_rsi_bear_s = pd.Series(True, index=df.index)

        sig_buy = cond_up & cond_macd_bull_s & cond_rsi_bull_s
        sig_sell = cond_down & cond_macd_bear_s & cond_rsi_bear_s
        signals = pd.Series(np.where(sig_buy, "BUY", np.where(sig_sell, "SELL", "HOLD")), index=df.index)

        metrics, bt = backtest_from_signals(df["Close"], signals, allow_short=allow_short, fee_bps=fee_bps)

        # KPI
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("CAGR", f"{metrics['cagr']*100:,.2f}%" if pd.notna(metrics['cagr']) else "N/A")
        kpi2.metric("Max Drawdown", f"{metrics['max_drawdown']*100:,.2f}%" if pd.notna(metrics['max_drawdown']) else "N/A")
        kpi3.metric("Sharpe", f"{metrics['sharpe']:.2f}" if pd.notna(metrics['sharpe']) else "N/A")
        kpi4.metric("Win Rate", f"{metrics['win_rate']:.1f}%")

        # Equity curve
        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(x=bt.index, y=bt["Equity"], name="Strategia", mode="lines"))
        fig_eq.add_trace(go.Scatter(x=bt.index, y=(bt["Close"] / bt["Close"].iloc[0]), name="Buy & Hold", mode="lines",
                                    line=dict(color="#888", dash="dot")))
        fig_eq.update_layout(template="plotly_dark", height=420, margin=dict(t=20,l=10,r=10,b=10),
                             yaxis_title="Equity (base=1.0)")
        st.plotly_chart(fig_eq, use_container_width=True)

        # Drawdown chart
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(x=bt.index, y=bt["Drawdown"], name="Drawdown", mode="lines", line=dict(color="#d9534f")))
        fig_dd.update_layout(template="plotly_dark", height=240, margin=dict(t=20,l=10,r=10,b=10),
                             yaxis_tickformat=".0%")
        st.plotly_chart(fig_dd, use_container_width=True)

        with st.expander("Dettagli numerici"):
            st.dataframe(bt.tail(10))
            st.json(metrics)