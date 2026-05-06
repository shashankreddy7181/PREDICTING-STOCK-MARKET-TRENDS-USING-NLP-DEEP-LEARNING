import os
import warnings

# Hide TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# Hide Scikit-Learn feature name warnings
warnings.filterwarnings("ignore", category=UserWarning)

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import timedelta

# --- Page Setup ---
st.set_page_config(page_title="AI Equity Terminal", layout="wide", initial_sidebar_state="expanded")

# --- High-Contrast & Theme-Aware Styling ---
st.markdown("""
    <style>
    [data-testid="stMetricValue"] { 
        font-size: 28px; 
        font-weight: 800;
        color: #1d4ed8; 
    }
    [data-testid="stMetricLabel"] { 
        font-size: 15px; 
        font-weight: 700;
        opacity: 0.9;
    }
    .stMetric {
        background-color: rgba(128, 128, 128, 0.05);
        border: 1px solid rgba(128, 128, 128, 0.2);
        padding: 15px;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Global Official Metrics (Synced to your table) ---
GLOBAL_METRICS = {
    "Baseline LSTM": {"Accuracy": 0.5265, "Precision": 0.5265, "Recall": 1.0000, "F1-Score": 0.6898, "AUC-ROC": 0.4777},
    "Hybrid NLP-Fusion": {"Accuracy": 0.5686, "Precision": 0.5783, "Recall": 0.6674, "F1-Score": 0.6197, "AUC-ROC": 0.5935},
    "NLP Tree Ensemble": {"Accuracy": 0.6284, "Precision": 0.6280, "Recall": 0.7216, "F1-Score": 0.6716, "AUC-ROC": 0.6756},
    "Multi-Model Ensemble": {"Accuracy": 0.6252, "Precision": 0.6223, "Recall": 0.7331, "F1-Score": 0.6732, "AUC-ROC": 0.6716}
}

# --- Data Loading ---
MODEL_DIR = 'saved_models'

@st.cache_data
def load_all():
    df = pd.read_csv(os.path.join(MODEL_DIR, 'df_final.csv'))
    date_col = 'Date' if 'Date' in df.columns else 'Price_Date'
    df[date_col] = pd.to_datetime(df[date_col])
    
    ps = joblib.load(os.path.join(MODEL_DIR, 'price_scaler.pkl'))
    ns = joblib.load(os.path.join(MODEL_DIR, 'nlp_scaler.pkl'))
    m_lstm = joblib.load(os.path.join(MODEL_DIR, 'model_a_lstm.pkl'))
    m_hybrid = joblib.load(os.path.join(MODEL_DIR, 'model_b_hybrid.pkl'))
    m_tree = joblib.load(os.path.join(MODEL_DIR, 'model_c_ensemble.pkl'))
    
    return df, ps, ns, m_lstm, m_hybrid, m_tree, date_col

df_full, ps, ns, m_lstm, m_hybrid, m_tree, DATE_COL = load_all()

# --- SIDEBAR: NAVIGATION ---
st.sidebar.title("📊 Terminal Navigation")
page = st.sidebar.radio("Select View", ["Market Dashboard", "Historical Data Audit", "Model Comparison"])

st.sidebar.divider()
st.sidebar.title("🔍 Strategy Controls")

ticker_col = next((c for c in ['Ticker', 'Symbol', 'stock'] if c in df_full.columns), None)
if ticker_col:
    selected_stock = st.sidebar.selectbox("Select Asset", sorted(df_full[ticker_col].unique()))
    df = df_full[df_full[ticker_col] == selected_stock].copy().sort_values(DATE_COL)
else:
    df = df_full.copy().sort_values(DATE_COL)

selected_model_name = st.sidebar.selectbox("Analysis Engine", list(GLOBAL_METRICS.keys()))

# --- Inference Engine ---
seq_len = 10
X_p = ps.transform(df[['Open', 'High', 'Low', 'Close', 'Volume']])
X_n = ns.transform(df[['News_Volume', 'News_Sentiment', 'NER_Score', 'TFIDF_Score']])

p_seq, n_feat, c_feat = [], [], []
for i in range(seq_len, len(X_p)):
    p_seq.append(X_p[i-seq_len:i])
    n_feat.append(X_n[i])
    c_feat.append(np.hstack([X_p[i-seq_len:i].flatten(), X_n[i]]))

p_seq, n_feat, c_feat = np.array(p_seq), np.array(n_feat), np.array(c_feat)
df_res = df.iloc[seq_len:].copy()

# Probabilities
prob_lstm = m_lstm.predict(p_seq).flatten()
prob_hybrid = m_hybrid.predict([p_seq, n_feat]).flatten()
prob_tree = m_tree.predict_proba(c_feat)[:, 1]
prob_ens = (0.2 * prob_lstm) + (0.3 * prob_hybrid) + (0.5 * prob_tree)

all_probs = {
    "Baseline LSTM": prob_lstm, "Hybrid NLP-Fusion": prob_hybrid,
    "NLP Tree Ensemble": prob_tree, "Multi-Model Ensemble": prob_ens
}

y_prob = all_probs[selected_model_name]
y_pred = (y_prob > 0.5).astype(int)

# --- MAIN CONTENT ---

if page == "Model Comparison":
    st.title("🏆 Strategy Benchmarking")
    st.markdown("Compare the official performance metrics of all trained models side-by-side.")

    # Create Comparison Table
    comparison_df = pd.DataFrame(GLOBAL_METRICS).T
    st.dataframe(comparison_df.style.highlight_max(axis=0, color='#1d4ed8'), use_container_width=True)

    st.divider()

    # Visual Comparison Chart
    st.subheader("📊 Performance Visualization")
    plot_df = comparison_df.reset_index().rename(columns={'index': 'Model'})
    
    fig_comp = px.bar(
        plot_df, x='Model', y=['Accuracy', 'AUC-ROC', 'F1-Score'],
        barmode='group',
        color_discrete_sequence=['#1d4ed8', '#10b981', '#f59e0b'],
        height=500
    )
    fig_comp.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", hovermode="x unified")
    st.plotly_chart(fig_comp, use_container_width=True)

    # Future Table
    st.subheader("🔮 Predictive Future Outlook")
    last_4_rows = df_res.tail(4)
    future_df = pd.DataFrame({
        "Analysis Date": last_4_rows[DATE_COL].dt.date.values,
        "Open Price": [f"${x:,.2f}" for x in last_4_rows['Open']],
        "Close Price": [f"${x:,.2f}" for x in last_4_rows['Close']],
        "Future Forecast": ["📈 UP" if p == 1 else "📉 DOWN" for p in y_pred[-4:]],
        "Confidence": [f"{prob:.1%}" if p==1 else f"{(1-prob):.1%}" for p, prob in zip(y_pred[-4:], y_prob[-4:])]
    })
    st.dataframe(future_df, use_container_width=True, hide_index=True)

    # REVERTED GRAPH (Line Chart)
    st.subheader("📊 Price Trend & AI Signals")
    fig = go.Figure()
    
    # Original Line Style
    fig.add_trace(go.Scatter(
        x=df_res[DATE_COL], y=df_res['Close'], 
        name='Market Price', 
        line=dict(color='#1d4ed8', width=2),
        customdata=np.stack((df_res['Open'], df_res['High'], df_res['Low']), axis=-1),
        hovertemplate="<b>Date: %{x}</b><br>Open: $%{customdata[0]:.2f}<br>Close: $%{y:.2f}<br>High: $%{customdata[1]:.2f}<br>Low: $%{customdata[2]:.2f}<extra></extra>"
    ))
    
    # Signals
    fig.add_trace(go.Scatter(
        x=df_res[y_pred == 1][DATE_COL], y=df_res[y_pred == 1]['Close'],
        mode='markers', name='AI Predicted UP',
        marker=dict(color='#10b981', size=10, symbol='triangle-up', line=dict(width=1, color='white'))
    ))
    fig.add_trace(go.Scatter(
        x=df_res[y_pred == 0][DATE_COL], y=df_res[y_pred == 0]['Close'],
        mode='markers', name='AI Predicted DOWN',
        marker=dict(color='#ef4444', size=10, symbol='triangle-down', line=dict(width=1, color='white'))
    ))
    
    fig.update_layout(height=500, hovermode="x unified", margin=dict(l=0, r=0, t=20, b=0))
    st.plotly_chart(fig, use_container_width=True)

elif page == "Historical Data Audit":
    st.title("🔢 Historical Audit")
    comp_df = pd.DataFrame({
        'Date': df_res[DATE_COL].dt.date,
        'Open Price': df_res['Open'],
        'Close Price': df_res['Close'],
        'Actual Move': df_res['Label'].map({1: '📈 UP', 0: '📉 DOWN'}),
        'AI Prediction': pd.Series(y_pred).map({1: '📈 UP', 0: '📉 DOWN'}).values,
        'Match': np.where(df_res['Label'] == y_pred, '✅ Correct', '❌ Error')
    })
    st.dataframe(comp_df.sort_values('Date', ascending=False), use_container_width=True, hide_index=True)