import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Page Configuration
st.set_page_config(
    page_title="Diamond Valuation System",
    page_icon="💎",
    layout="centered"
)

# Custom Styling (Optional)
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. Resource Loading
@st.cache_resource
def load_model_assets():
    try:
        model = joblib.load('model_xgb.pkl')
        scaler = joblib.load('scaler_xgb.pkl')
        return model, scaler, None
    except Exception as e:
        return None, None, str(e)

model, scaler, error = load_model_assets()

# 3. Sidebar / Header
st.title("💎 Diamond Price Analytics")
st.markdown("""
Sistem estimasi harga berlian menggunakan algoritma **XGBoost Regression**. 
Masukkan spesifikasi teknis di bawah ini untuk mendapatkan penilaian instan.
""")
st.divider()

if error:
    st.error(f"Sistem gagal memuat modul prediksi: {error}")
    st.stop()

# 4. Input Sections
st.subheader("📍 Spesifikasi Utama")
col1, col2 = st.columns(2)

with col1:
    carat = st.number_input("Carat Weight (Berat)", min_value=0.1, max_value=5.0, value=0.7, step=0.01, help="Berat berlian dalam karat.")
    cut = st.selectbox("Cut Quality", ["Ideal", "Premium", "Very Good", "Good", "Fair"])
    color = st.selectbox("Color Grade", ["D", "E", "F", "G", "H", "I", "J"], help="D adalah kualitas warna tertinggi.")
    clarity = st.selectbox("Clarity Grade", ["IF", "VVS1", "VVS2", "VS1", "VS2", "SI1", "SI2", "I1"])

with col2:
    depth = st.number_input("Depth %", 43.0, 79.0, 61.8)
    table = st.number_input("Table %", 43.0, 95.0, 57.0)
    x = st.number_input("Length (x) mm", 0.1, 11.0, 5.7)
    y = st.number_input("Width (y) mm", 0.1, 58.0, 5.7)
    z = st.number_input("Depth (z) mm", 0.1, 31.0, 3.5)

# Mapping Categorical Data (Sesuai dengan Label Encoding saat training)
cut_map = {"Fair": 0, "Good": 1, "Very Good": 2, "Premium": 3, "Ideal": 4}
color_map = {"J": 0, "I": 1, "H": 2, "G": 3, "F": 4, "E": 5, "D": 6}
clarity_map = {"I1": 0, "SI2": 1, "SI1": 2, "VS2": 3, "VS1": 4, "VVS2": 5, "VVS1": 6, "IF": 7}

# 5. Prediction Logic
st.divider()
if st.button("📊 Hitung Nilai Estimasi"):
    # Urutan fitur: carat, cut, color, clarity, depth, table, x, y, z
    features = pd.DataFrame([[
        carat, cut_map[cut], color_map[color], clarity_map[clarity],
        depth, table, x, y, z
    ]], columns=['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'x', 'y', 'z'])

    with st.spinner('Menganalisis data pasar...'):
        try:
            # Scaling & Predicting
            scaled_features = scaler.transform(features)
            prediction = model.predict(scaled_features)
            
            # Display Results
            st.success("Analisis Berhasil Selesai!")
            
            # Formatting Price
            price = max(0, prediction[0])
            
            # Visualisasi Hasil dengan Metric
            c1, c2, c3 = st.columns([1,2,1])
            with c2:
                st.metric(label="Estimasi Harga (USD)", value=f"${price:,.2f}")
                
            st.info(f"**Ringkasan:** Berlian dengan berat **{carat} karat** dan kualitas **{cut}** diperkirakan bernilai sekitar **${price:,.2f}** di pasar saat ini.")
            
        except Exception as e:
            st.error(f"Terjadi kesalahan teknis saat prediksi: {e}")

# Footer
st.divider()
st.caption("© 2024 Diamond Valuation System | Powered by XGBoost & Streamlit Cloud")