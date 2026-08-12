import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)

# ============================================
# CONFIG
# ============================================
st.set_page_config(
    page_title="Fraud Detection Dashboard - DSC 2026",
    page_icon="🏦",
    layout="wide",
)

DATA_PATH = "enriched_data.csv"

ACCENT_TEAL = "#2DD4BF"
ACCENT_PURPLE = "#A78BFA"
ACCENT_BLUE = "#38BDF8"
BG_CARD = "#131826"

# ============================================
# STYLING
# ============================================
custom_css = """
<style>
:root {
    --bg-dark: #0B0F19;
    --bg-card: #131826;
    --accent-teal: #2DD4BF;
    --accent-purple: #A78BFA;
    --accent-blue: #38BDF8;
}
.stApp {
    background: radial-gradient(circle at 15% 10%, rgba(45,212,191,0.08), transparent 40%),
                radial-gradient(circle at 85% 90%, rgba(167,139,250,0.08), transparent 40%),
                var(--bg-dark) !important;
    font-family: 'Inter', 'Segoe UI', sans-serif !important;
}
h1 {
    background: linear-gradient(90deg, var(--accent-teal), var(--accent-blue), var(--accent-purple));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700 !important;
}
h3 { color: #E5E7EB !important; }
[data-testid="stMetric"], [data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(19, 24, 38, 0.75) !important;
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 16px !important;
    box-shadow: 0 4px 24px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.03) !important;
    padding: 10px !important;
}
[data-testid="stMetricValue"] { color: var(--accent-teal) !important; }
table { background: transparent !important; color: #E5E7EB !important; }
thead th {
    background: rgba(45,212,191,0.08) !important;
    color: var(--accent-teal) !important;
}
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: var(--bg-dark); }
::-webkit-scrollbar-thumb { background: rgba(45,212,191,0.4); border-radius: 8px; }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)


def style_fig(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BG_CARD,
        plot_bgcolor=BG_CARD,
        font=dict(color="#E5E7EB", family="Inter, sans-serif"),
        title_font=dict(color="#F9FAFB", size=15),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=30, r=20, t=50, b=30),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.08)", zerolinecolor="rgba(255,255,255,0.1)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.08)", zerolinecolor="rgba(255,255,255,0.1)")
    return fig


# ============================================
# LOAD DATA (cached so it doesn't reload on every filter change)
# ============================================
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["txn_date"] = pd.to_datetime(df["txn_date"])
    return df


try:
    df = load_data()
except FileNotFoundError:
    st.error(
        f"❌ File `{DATA_PATH}` tidak ditemukan. Pastikan file ini ada di root repo "
        "(satu folder yang sama dengan streamlit_app.py)."
    )
    st.stop()

RISK_OPTIONS = sorted(df["risk_level"].dropna().unique().tolist())
CAT_OPTIONS = sorted(df["merchant_category"].dropna().unique().tolist())
MIN_DATE = df["txn_date"].min().date()
MAX_DATE = df["txn_date"].max().date()

# ============================================
# HEADER
# ============================================
st.markdown("# 🏦 Bank Fraud Detection Dashboard")
st.markdown("DSC 2026 Hackathon — Fraud Detection Analytics")

# ============================================
# FILTERS
# ============================================
with st.container():
    c1, c2, c3 = st.columns([2, 2, 2])
    with c1:
        risk_sel = st.multiselect("Risk Level", RISK_OPTIONS, default=RISK_OPTIONS)
    with c2:
        cat_sel = st.multiselect("Merchant Category", CAT_OPTIONS, default=CAT_OPTIONS)
    with c3:
        date_range = st.date_input(
            "Rentang Tanggal", value=(MIN_DATE, MAX_DATE), min_value=MIN_DATE, max_value=MAX_DATE
        )

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = MIN_DATE, MAX_DATE

f = df[
    df["risk_level"].isin(risk_sel)
    & df["merchant_category"].isin(cat_sel)
    & (df["txn_date"] >= pd.Timestamp(start_date))
    & (df["txn_date"] <= pd.Timestamp(end_date))
]

if len(f) == 0:
    st.warning("⚠️ Tidak ada data untuk kombinasi filter ini.")
    st.stop()

# ============================================
# KPI ROW
# ============================================
st.markdown("### 📊 Ringkasan")
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Transaksi", f"{len(f):,}")
k2.metric("Predicted Fraud", f"{int(f['predicted_fraud'].sum()):,}")
k3.metric("Fraud Rate (Predicted)", f"{f['predicted_fraud'].mean()*100:.2f}%")
k4.metric("Avg Fraud Probability", f"{f['fraud_probability'].mean():.3f}")
k5.metric("Life Events Terdeteksi", f"{int(f['is_life_event'].sum()):,}")

# ============================================
# MODEL PERFORMANCE + CONFUSION MATRIX
# ============================================
y_true = f["is_fraud"]
y_pred = f["predicted_fraud"]
y_proba = f["fraud_probability"]

st.markdown("### 🎯 Performa Model (pada data terfilter)")
if y_true.nunique() > 1:
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_proba)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Precision", f"{prec:.3f}")
    m2.metric("Recall", f"{rec:.3f}")
    m3.metric("F1-Score", f"{f1:.3f}")
    m4.metric("ROC-AUC", f"{auc:.3f}")

    cm = confusion_matrix(y_true, y_pred)
    fig_cm = px.imshow(
        cm, text_auto=True,
        x=["Predicted: Non-Fraud", "Predicted: Fraud"],
        y=["Actual: Non-Fraud", "Actual: Fraud"],
        color_continuous_scale=[[0, BG_CARD], [1, ACCENT_TEAL]],
        title="Confusion Matrix (Actual vs Predicted)"
    )
else:
    st.info("⚠️ Data terfilter hanya punya 1 kelas (fraud/non-fraud), metrik tidak bisa dihitung.")
    fig_cm = px.scatter(title="Confusion Matrix tidak tersedia untuk filter ini")
fig_cm = style_fig(fig_cm)

# ============================================
# CHARTS ROW 1
# ============================================
col1, col2 = st.columns(2)
with col1:
    risk_counts = f["risk_level"].value_counts().reset_index()
    risk_counts.columns = ["risk_level", "count"]
    fig_pie = style_fig(px.pie(
        risk_counts, names="risk_level", values="count", color="risk_level",
        color_discrete_map={"Low": ACCENT_TEAL, "Medium": ACCENT_BLUE, "High": "#F472B6"},
        title="Distribusi Risk Level", hole=0.45
    ))
    st.plotly_chart(fig_pie, use_container_width=True)
with col2:
    fig_hist = style_fig(px.histogram(
        f, x="fraud_probability", nbins=40, color="predicted_fraud",
        title="Distribusi Fraud Probability",
        color_discrete_sequence=[ACCENT_PURPLE, "#F472B6"]
    ))
    st.plotly_chart(fig_hist, use_container_width=True)

# ============================================
# CHARTS ROW 2
# ============================================
col3, col4 = st.columns(2)
with col3:
    cat_fraud = (
        f.groupby("merchant_category")["predicted_fraud"]
        .sum().sort_values(ascending=False).reset_index()
    )
    fig_bar = style_fig(px.bar(
        cat_fraud, x="merchant_category", y="predicted_fraud",
        title="Predicted Fraud per Merchant Category",
        color_discrete_sequence=[ACCENT_TEAL]
    ))
    st.plotly_chart(fig_bar, use_container_width=True)
with col4:
    ts = (
        f.groupby(f["txn_date"].dt.date)
        .agg(total=("amount", "count"), fraud=("predicted_fraud", "sum"))
        .reset_index()
    )
    fig_line = style_fig(px.line(
        ts, x="txn_date", y=["total", "fraud"], title="Transaksi & Fraud per Hari",
        color_discrete_sequence=[ACCENT_BLUE, "#F472B6"]
    ))
    st.plotly_chart(fig_line, use_container_width=True)

# ============================================
# CHARTS ROW 3
# ============================================
col5, col6 = st.columns(2)
with col5:
    hourly = (
        f.groupby("txn_hour")
        .agg(total=("amount", "count"), fraud=("predicted_fraud", "sum"))
        .reset_index()
    )
    fig_hour = style_fig(px.bar(
        hourly, x="txn_hour", y="fraud",
        title="Predicted Fraud per Jam Transaksi (0-23)",
        labels={"txn_hour": "Jam", "fraud": "Jumlah Predicted Fraud"},
        color_discrete_sequence=[ACCENT_PURPLE]
    ))
    st.plotly_chart(fig_hour, use_container_width=True)
with col6:
    st.plotly_chart(fig_cm, use_container_width=True)

# ============================================
# TABLES
# ============================================
st.markdown("### 🚨 Alert: Transaksi High Risk")
high_risk = f[f["risk_level"] == "High"].sort_values("fraud_probability", ascending=False)
alert_cols = [c for c in [
    "card_txn_id", "customer_id", "name", "amount", "merchant_category",
    "txn_date", "fraud_probability", "risk_level"
] if c in high_risk.columns]
st.dataframe(high_risk[alert_cols], use_container_width=True)

st.markdown("### 🎉 Life Event Detection (Transaksi Besar di Kategori Tertentu)")
life_events = f[f["is_life_event"] == True].sort_values("amount", ascending=False)
life_cols = [c for c in [
    "card_txn_id", "customer_id", "name", "amount", "merchant_category",
    "txn_date", "avg_amount", "risk_level"
] if c in life_events.columns]
st.dataframe(life_events[life_cols], use_container_width=True)

st.markdown("### 📄 Data Transaksi (maks 500 baris ditampilkan)")
st.dataframe(f.head(500), use_container_width=True)

# ============================================
# DOWNLOAD
# ============================================
csv_bytes = f.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️ Download Data Terfilter (CSV)",
    data=csv_bytes,
    file_name="filtered_fraud_data.csv",
    mime="text/csv",
)
