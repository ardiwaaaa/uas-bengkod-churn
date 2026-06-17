import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==============================================================================
# CONFIG & THEME CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="Customer Churn Prediction App",
    page_icon="📊",
    layout="wide"
)

# Judul Utama Dashboard
st.write("""
# 📊 Customer Churn Prediction Dashboard
### Aplikasi Prediksi Retensi Pelanggan — UAS Bengkel Koding Data Science
Aplikasi ini memprediksi apakah seorang pelanggan berpotensi *churn* (berhenti menggunakan layanan) atau tetap setia berdasarkan profil demografi dan riwayat transaksinya menggunakan model Machine Learning optimal.
---
""")

# ==============================================================================
# LOAD MODEL & PREPROCESSOR PIEPLINE
# ==============================================================================
@st.cache_resource
def load_ml_components():
    try:
        model = joblib.load('best_churn_model.pkl')
        preprocessor = joblib.load('preprocessor.pkl')
        return model, preprocessor
    except FileNotFoundError:
        st.error("⚠️ File 'best_churn_model.pkl' atau 'preprocessor.pkl' tidak ditemukan di repositori GitHub ini. Pastikan Anda sudah mengunggah kedua file tersebut.")
        return None, None

model, preprocessor = load_ml_components()

# ==============================================================================
# SIDEBAR INPUT FORM
# ==============================================================================
if model and preprocessor:
    st.sidebar.header("📥 Input Karakteristik Pelanggan")
    
    with st.sidebar.form(key='churn_form'):
        st.subheader("👤 Profil Demografi")
        gender = st.selectbox("Jenis Kelamin (Gender)", options=["Male", "Female"])
        country = st.selectbox("Negara Asal (Country)", options=["United States", "Canada", "United Kingdom", "Germany", "France"])
        device_type = st.selectbox("Perangkat Utama (Device Type)", options=["Mobile", "Desktop", "Tablet"])
        payment_method = st.selectbox("Metode Pembayaran", options=["Credit Card", "PayPal", "Bank Transfer", "Electronic Wallet"])
        
        st.markdown("---")
        st.subheader("📈 Aktivitas & Transaksi")
        recency = st.slider("Recency (Hari sejak transaksi terakhir)", min_value=0, max_value=365, value=30)
        frequency = st.number_input("Frequency (Total jumlah transaksi)", min_value=1, max_value=100, value=5)
        total_spent = st.number_input("Total Spent (Total nominal pengeluaran)", min_value=0.0, value=500.0)
        lifetime_value = st.number_input("Lifetime Value (CLV)", min_value=0.0, value=1500.0)
        satisfaction_score = st.slider("Skor Kepuasan Pelanggan (1-5)", min_value=1, max_value=5, value=4)
        delivery_delay_days = st.number_input("Rata-rata Keterlambatan Pengiriman (Hari)", min_value=0, max_value=30, value=2)
        
        # Tombol Submit Form
        submit_button = st.form_submit_button(label='💡 Jalankan Prediksi Churn')

    # ==============================================================================
    # PREDICTION LOGIC & OUTPUT DISPLAY
    # ==============================================================================
    if submit_button:
        # 1. Mengubah input user menjadi DataFrame (sesuai nama kolom dataset asli)
        input_data = pd.DataFrame([{
            'gender': gender,
            'country': country,
            'device_type': device_type,
            'payment_method': payment_method,
            'recency': recency,
            'frequency': frequency,
            'total_spent': total_spent,
            'lifetime_value': lifetime_value,
            'satisfaction_score': satisfaction_score,
            'delivery_delay_days': delivery_delay_days
        }])
        
        # Menampilkan Ringkasan Data Input di Area Utama
        st.subheader("📋 Ringkasan Data Pelanggan Baru")
        st.dataframe(input_data)
        
        try:
            # 2. Transformasi data menggunakan preprocessor pipa (Scaling & Encoding)
            input_processed = preprocessor.transform(input_data)
            
            # 3. Prediksi menggunakan model optimal hasil Minggu 3
            prediction = model.predict(input_processed)[0]
            prediction_proba = model.predict_proba(input_processed)[0]
            
            st.markdown("---")
            st.subheader("🔮 Hasil Analisis & Prediksi Model")
            
            # Membagi layout visual menjadi 2 kolom
            col1, col2 = st.columns(2)
            
            with col1:
                if prediction == 1:
                    st.error("🚨 HASIL PREDIKSI: BERPOTENSI CHURN (PERGI)")
                    st.metric(label="Status Risiko", value="Tinggi (Churn)", delta="Perlu Intervensi Segera", delta_color="inverse")
                else:
                    st.success("✅ HASIL PREDIKSI: TETAP BERLANGGANAN (LOYAL)")
                    st.metric(label="Status Risiko", value="Rendah (Loyal)", delta="Pertahankan Layanan", delta_color="normal")
            
            with col2:
                st.write("**Keyakinan Model (Probability Score):**")
                st.write(f"• Probabilitas Tetap Setia (0): `{prediction_proba[0] * 100:.2f}%`")
                st.write(f"• Probabilitas Akan Churn (1) : `{prediction_proba[1] * 100:.2f}%`")
                
                # Visualisasi tingkat risiko churn dengan progress bar
                st.progress(float(prediction_proba[1]))
                
            # Rekomendasi Keputusan Bisnis Berdasarkan Hasil Prediksi
            st.markdown("### 💡 Rekomendasi Tindakan Bisnis:")
            if prediction == 1:
                st.info("👉 **Strategi Retensi:** Kirimkan email penawaran khusus, berikan voucher diskon loyalitas, atau hubungi customer service untuk survei kepuasan guna mencegah mereka benar-benar pergi.")
            else:
                st.info("👉 **Strategi Retensi:** Masukkan pelanggan ke program loyalitas reguler dan tawarkan produk pelengkap (cross-selling/upselling) karena retensi mereka terpantau stabil.")
                
        except Exception as e:
            st.error(f"❌ Terjadi kesalahan dalam pemrosesan data: {e}")
            st.warning("Catatan: Pastikan urutan dan nama kolom input data di atas sama persis dengan nama fitur yang dilatih pada model Minggu 2.")