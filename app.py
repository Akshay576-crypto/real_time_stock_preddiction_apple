
import streamlit as st
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Stock Price Prediction", layout="wide")

st.title("Real-Time Stock Price Prediction System")
st.write("Predict next day stock price with Buy / Hold / Sell signal")


stock_symbol = st.text_input("Enter Stock Symbol", "AAPL")

if st.button("Predict"):
    # Download data
    data = yf.download(stock_symbol, period="5y")

    if data.empty:
        st.error("No data found. Please check stock symbol.")
    else:
        # Keep required columns
        df = data[["Open", "High", "Low", "Volume", "Close"]].copy()

        # Target = next day's close price
        df["Target"] = df["Close"].shift(-1)
        df.dropna(inplace=True)

        # Features and target
        X = df[["Open", "High", "Low", "Volume", "Close"]]
        y = df["Target"]

        
        model = LinearRegression()
        model.fit(X, y)

        # Latest row prediction
        latest_data = X.tail(1)
        predicted_price = float(model.predict(latest_data)[0])
        current_price = float(latest_data["Close"].iloc[0].item())

        # Buy / Sell Logic
        if predicted_price > current_price:
            signal = "BUY"
        elif predicted_price < current_price:
            signal = "SELL"
        else:
            signal = "HOLD"

        
        st.subheader("Prediction Result")
        st.write(f"Current Price: {round(current_price, 2)}")
        st.write(f"Predicted Next Day Price: {round(predicted_price, 2)}")
        st.write(f"Suggested Action: {signal}")

        
        st.subheader("Recent Stock Data")
        st.dataframe(df.tail())

        # Line chart
        st.subheader("Closing Price Trend")
        st.line_chart(df["Close"])
