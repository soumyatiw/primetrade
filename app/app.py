import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from PIL import Image

# Set page configuration
st.set_page_config(page_title="Primetrade Sentiment Analysis", layout="wide")

# App title and description
st.title("Primetrade Sentiment Analysis")
st.markdown("""
A data science dashboard exploring the relationship between market sentiment and trader behavior on Hyperliquid.
""")

# Custom CSS for Light UI and KPI Borders
st.markdown("""
<style>
/* KPI Metric Cards styling */
[data-testid="stMetric"] {
    background-color: #FFFFFF;
    border: 1px solid #E0E0E0;
    border-radius: 8px;
    padding: 15px 20px;
    box-shadow: 0px 2px 5px rgba(0, 0, 0, 0.05);
}

/* Lighter font for metric labels */
[data-testid="stMetricLabel"] {
    color: #666666;
    font-weight: 500;
}

/* Better spacing around expanders and columns */
.css-1544g2n {
    padding-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

# Define paths
DATA_DIR = os.path.join(os.path.dirname(__file__), '../data')
CHARTS_DIR = os.path.join(os.path.dirname(__file__), '../charts')
MODELS_DIR = os.path.join(os.path.dirname(__file__), '../models')

# -----------------------------------------------------------------------------
# Data Loading & Caching
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    # Load raw data
    sentiment_path = os.path.join(DATA_DIR, 'sentiment.csv')
    traders_path = os.path.join(DATA_DIR, 'traders.csv')
    
    df_sentiment = pd.read_csv(sentiment_path)
    df_trades = pd.read_csv(traders_path)
    
    # Preprocess sentiment
    df_sentiment['date'] = pd.to_datetime(df_sentiment['date']).dt.date
    
    # Preprocess trades
    df_trades['trade_date'] = pd.to_datetime(df_trades['Timestamp IST'], format='%d-%m-%Y %H:%M').dt.date
    df_trades.rename(columns={
        'Account': 'account', 'Execution Price': 'exec_price', 
        'Size USD': 'size_usd', 'Side': 'side', 'Closed PnL': 'closed_pnl'
    }, inplace=True)
    
    # Create closed trades subset
    df_closed = df_trades[df_trades['closed_pnl'] != 0].copy()
    
    # Merge
    trade_dates = set(df_trades['trade_date'].unique())
    df_sentiment_trim = df_sentiment[df_sentiment['date'].isin(trade_dates)].copy()
    
    df_closed = df_closed.merge(df_sentiment_trim, left_on='trade_date', right_on='date', how='inner')
    
    return df_sentiment_trim, df_trades, df_closed

df_sentiment, df_trades, df_closed = load_data()

# -----------------------------------------------------------------------------
# Model Loading
# -----------------------------------------------------------------------------
@st.cache_resource
def load_model():
    model_path = os.path.join(MODELS_DIR, 'rf_model.pkl')
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        return None

rf_model = load_model()

# -----------------------------------------------------------------------------
# Tabs Layout
# -----------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["Market Overview", "Sentiment Insights", "Profit Predictor"])

# ==========================================
# TAB 1: Market Overview
# ==========================================
with tab1:
    st.header("Dataset Overview")
    
    # Top-level metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    total_trades = len(df_trades)
    total_traders = df_trades['account'].nunique()
    start_date = df_trades['trade_date'].min()
    end_date = df_trades['trade_date'].max()
    
    # Calculate overall win rate
    win_rate = (df_closed['closed_pnl'] > 0).sum() / len(df_closed) * 100
    
    col1.metric("Total Executions", f"{total_trades:,}")
    col2.metric("Unique Traders", f"{total_traders}")
    col3.metric("Start Date", f"{start_date}")
    col4.metric("End Date", f"{end_date}")
    col5.metric("Overall Win Rate", f"{win_rate:.1f}%")
    
    st.divider()
    
    col_a, col_b = st.columns([1, 2])
    
    with col_a:
        st.subheader("Sentiment Distribution")
        sentiment_counts = df_closed['classification'].value_counts()
        st.bar_chart(sentiment_counts)
        
    with col_b:
        st.subheader("Sample Trade Data")
        st.dataframe(df_closed[['account', 'trade_date', 'side', 'size_usd', 'closed_pnl', 'classification']].head(100))

# ==========================================
# TAB 2: Sentiment Insights
# ==========================================
with tab2:
    st.header("How Sentiment Impacts Trading")
    st.markdown("These charts were generated during our EDA and highlight the key relationships between the Fear & Greed index and trader behavior.")
    
    def display_chart(filename, title, insight):
        filepath = os.path.join(CHARTS_DIR, filename)
        if os.path.exists(filepath):
            st.subheader(title)
            st.image(Image.open(filepath), use_column_width=True)
            st.info(f"**Key Insight:** {insight}")
        else:
            st.warning(f"Chart {filename} not found in charts directory.")

    col1, col2 = st.columns(2)
    
    with col1:
        display_chart(
            "chart1_pnl_distribution.png", 
            "1. PnL Distribution by Sentiment", 
            "Extreme Greed and Fear days show wider spreads in PnL, indicating more dispersion in outcomes compared to Neutral days."
        )
        st.divider()
        display_chart(
            "chart3_fee_rate.png", 
            "3. Average Fee Rate (Leverage Proxy)", 
            "Fee rates are highest during Fear and Extreme Fear days, indicating that traders incur greater per-dollar costs (wider spreads, aggressive execution) in bearish sentiment."
        )
        
    with col2:
        display_chart(
            "chart2_win_rate.png", 
            "2. Win Rate by Sentiment", 
            "Win rates peak during Extreme Greed (~89%) and decline toward Extreme Fear (~76%). However, variance across accounts is high."
        )
        st.divider()
        display_chart(
            "chart4_trade_frequency.png", 
            "4. Trade Frequency", 
            "Extreme Fear days see by far the highest trade frequency (~1,529 trades/day), confirming that fear-driven markets trigger a surge in execution activity."
        )

# ==========================================
# TAB 3: Profitability Predictor
# ==========================================
with tab3:
    st.header("Next-Day Profitability Predictor")
    st.markdown("Use the Random Forest model to predict if tomorrow will be a profitable day based on today's inputs.")
    
    if rf_model is None:
        st.error("Model file `rf_model.pkl` not found. Please ensure it is saved in the `models/` directory.")
    else:
        col_input, col_output = st.columns([1, 1])
        
        with col_input:
            st.subheader("Today's Inputs")
            
            # Sentiment Inputs
            sentiment_class = st.selectbox("Today's Market Sentiment", ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"])
            sentiment_map = {'Extreme Fear': 0, 'Fear': 1, 'Neutral': 2, 'Greed': 3, 'Extreme Greed': 4}
            sentiment_encoded = sentiment_map[sentiment_class]
            
            # Since we don't have the exact sentiment value for the input, we approximate based on class for the 'sentiment_value' feature
            approx_val_map = {'Extreme Fear': 15, 'Fear': 35, 'Neutral': 50, 'Greed': 65, 'Extreme Greed': 85}
            sentiment_value = approx_val_map[sentiment_class]
            
            # Behavior Inputs
            trade_count = st.number_input("Number of Trades Today", min_value=1, max_value=5000, value=100)
            avg_trade_size = st.number_input("Average Trade Size (USD)", min_value=10.0, max_value=100000.0, value=5000.0)
            long_ratio = st.slider("Long Ratio (% BUY Trades)", min_value=0.0, max_value=1.0, value=0.5)
            
            # Performance Inputs
            num_closed = st.number_input("Number of Closed Trades Today", min_value=0, max_value=5000, value=50)
            prev_daily_pnl = st.number_input("Yesterday's PnL (USD)", value=150.0)
            
            # Dummy values for remaining features required by model
            # FEATURES = ['sentiment_value', 'sentiment_encoded', 'sentiment_3d_ma', 'sentiment_change',
            # 'trade_count', 'total_volume', 'avg_trade_size', 'avg_fee', 'long_ratio',
            # 'prev_daily_pnl', 'prev_win_rate', 'prev_sentiment_value', 'pnl_3d_ma', 'num_closed']
            
            # We estimate the required remaining features to keep the UI simple
            total_volume = trade_count * avg_trade_size
            avg_fee = 0.0005 # rough average
            sentiment_3d_ma = sentiment_value # Assume stable
            sentiment_change = 0.0 # Assume no change
            prev_win_rate = 0.6 # Assume 60%
            prev_sentiment_value = sentiment_value
            pnl_3d_ma = prev_daily_pnl # Assume stable PnL trend
            
        with col_output:
            st.subheader("Prediction")
            
            # Create feature array in exact order model expects
            features = np.array([[
                sentiment_value, sentiment_encoded, sentiment_3d_ma, sentiment_change,
                trade_count, total_volume, avg_trade_size, avg_fee, long_ratio,
                prev_daily_pnl, prev_win_rate, prev_sentiment_value, pnl_3d_ma,
                num_closed
            ]])
            
            if st.button("Predict Tomorrow's Outcome"):
                prediction = rf_model.predict(features)[0]
                probability = rf_model.predict_proba(features)[0]
                
                st.write("")
                if prediction == 1:
                    st.success(f"Prediction: Profit Day (Confidence: {probability[1]*100:.1f}%)")
                else:
                    st.error(f"Prediction: Loss Day (Confidence: {probability[0]*100:.1f}%)")
                    
            st.divider()
            st.subheader("Why?")
            st.markdown("Here is the overall feature importance calculated from the Random Forest model across all data:")
            
            feat_imp_path = os.path.join(CHARTS_DIR, "feature_importance.png")
            if os.path.exists(feat_imp_path):
                st.image(Image.open(feat_imp_path), use_column_width=True)
            else:
                st.info("Train the model in the notebook to generate the feature importance chart.")
