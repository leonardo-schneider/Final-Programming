import streamlit as st
import yfinance as yf

st.title("Financial Dashboard Project")

ticker = st.sidebar.text_input("Enter a Stock Ticker:", "AAPL")

if st.sidebar.button("Analyze Stock"):

    st.write(f"Fetching data for: {ticker}...")
    stock = yf.Ticker(ticker)

    history = stock.history(period="1y")

    if history.empty:
        st.error(f"Could not find data for {ticker}. Check the spelling.")
    else:
        current_price = history["Close"].iloc[-1]
        st.metric(label="Current Price", value=f"${current_price:.2f}")

        st.subheader("Price History (1 Year)")
        st.line_chart(history["Close"])

        st.subheader("Raw Data")
        st.dataframe(history.tail())

        st.subheader("Company Profile")
        summary = stock.info.get("longBusinessSummary", "No summary available.")
        st.info(summary)