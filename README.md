# 🏦 Bank Fraud Detection Dashboard — DSC 2026

Dashboard interaktif untuk deteksi fraud transaksi kartu, dibuat dengan Streamlit,
Plotly, dan model XGBoost (dengan SMOTE + threshold tuning).

## 📂 Struktur repo yang dibutuhkan

```
nama-repo/
├── streamlit_app.py       # Kode dashboard (file ini)
├── requirements.txt       # Dependencies
├── enriched_data.csv      # Data hasil prediksi dari Step 5A di Colab
└── README.md
```

## 🚀 Cara deploy ke Streamlit Community Cloud (gratis, permanen)

### 1. Ambil file `enriched_data.csv` dari Colab
Di Colab, download filenya:
```python
from google.colab import files
files.download("/content/ml_ready/enriched_data.csv")
```

### 2. Buat repo baru di GitHub
- Buat repo baru (bisa public, biar gratis di Streamlit Cloud)
- Upload 3 file: `streamlit_app.py`, `requirements.txt`, `enriched_data.csv`
  (drag-and-drop langsung di web GitHub juga bisa, tidak perlu command line)

### 3. Deploy di Streamlit Community Cloud
1. Buka https://share.streamlit.io dan login pakai akun GitHub
2. Klik **"New app"**
3. Pilih repo, branch (biasanya `main`), dan main file path: `streamlit_app.py`
4. Klik **Deploy**

Setelah build selesai (biasanya 1-3 menit), kamu akan dapat URL permanen seperti:
```
https://nama-app-kamu.streamlit.app
```

URL ini bisa dibuka siapa saja, kapan saja, tanpa perlu run notebook — cocok untuk
dilampirkan di LinkedIn atau CV sebagai link portofolio.

## 🔄 Update dashboard di kemudian hari
Setiap kali kamu push perubahan ke branch `main` di GitHub (misal update
`enriched_data.csv` dengan data terbaru, atau ubah kode dashboard), Streamlit
Cloud otomatis re-deploy app-nya — tidak perlu setup ulang.

## 📝 Catatan untuk LinkedIn
Beberapa hal yang bisa kamu highlight saat share project ini:
- Pipeline end-to-end: raw data → feature engineering → model XGBoost (SMOTE +
  threshold tuning berbasis precision-recall curve, bukan angka arbitrary) →
  dashboard interaktif
- Fitur dashboard: filter risk level/kategori/tanggal, confusion matrix live,
  deteksi "life event" dari pola transaksi, alert transaksi high-risk
- Stack: Python, Pandas, Scikit-learn, XGBoost, Plotly, Streamlit
