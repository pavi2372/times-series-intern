# Save this as app.py
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import numpy as np

st.set_page_config(page_title="Stock Dashboard", layout="wide")
st.title("📈 Stock Price Dashboard")

# ✅ Load your dataset (update the path if needed)
df = pd.read_csv("dataset.csv", parse_dates=["Date"], index_col="Date")

# ✅ Show actual columns
st.sidebar.write("Columns in dataset:", df.columns.tolist())

# ✅ Use 'close' or 'Close' depending on your file
col_name = "close" if "close" in df.columns else "Close"

# ✅ Plot actual stock prices
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df[col_name], mode='lines', name='Actual Close'))

# ✅ Customize plot
fig.update_layout(
    title="Stock Price",
    xaxis_title="Date",
    yaxis_title="Price",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)

