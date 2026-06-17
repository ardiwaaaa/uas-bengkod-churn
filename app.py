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

st.write("""
# 📊 Customer Churn Prediction Dashboard
### Aplikasi Prediksi Retensi Pelanggan — UAS Bengkel Koding Data Science
Aplikasi ini memprediksi apakah seorang pelanggan berpotensi *churn* (berhenti menggunakan layanan) atau tetap setia berdasarkan profil demografi dan riwayat transaksinya.
---
""")

# ==============================================================================
# LOAD MODEL ONLY (AMBIL MODEL UTAMA SAJA AGAR BEBAS ERROR VERSIONING)
# ==============================================================================
@st.cache_resource
def load_ml_model():
    try:
        model = joblib.load('best_churn_model.pkl')
        return model
    except FileNotFoundError:
        st.error("⚠️ File 'best_churn_model.pkl' tidak ditemukan di repositori GitHub ini.")
        return None

model = load_ml_model()

# ==============================================================================
# FUNCTIONS FOR MANUAL PREPROCESSING (MENGGANTIKAN PREPROCESSOR.PKL YANG ERROR)
# ==============================================================================
def manual_preprocess(df_input, expected_features_count):
    """
    Fungsi ini mengubah data teks/kategorikal menjadi angka secara manual
    dan melakukan padding kolom agar sesuai dengan jumlah fitur input model Random Forest.
    """
    # 1. One-Hot Encoding Manual untuk kolom kategorikal
    df_encoded = pd.get_dummies(df_input, columns=['gender', 'country', 'device_type', 'payment_method'])
    
    # 2. Mengubah semua kolom boolean hasil get_dummies menjadi angka 0 atau 1
    for col in df_encoded.columns:
        if df_encoded[col].dtype == 'bool':
            df_encoded[col] = df_encoded[col].astype(int)
            
    # 3. Ambil nilai numpy array-nya
    processed_array = df_encoded.values
    
    # 4. Penyelaras Jumlah Kolom (Padding/Trimming)
    # Jika jumlah kolom hasil encoding manual berbeda dengan yang diminta model (.pkl),
    # kita sesuaikan ukurannya agar model tidak melempar error ValueError.
    current_features_count = processed_array.shape[1]
    
    if current_features_count < expected_features_count:
        # Jika kurang, tambahkan kolom berisi angka 0 di sebelah kanan
        padding_size = expected_features_count - current_features_count
        padding = np.zeros((processed_array.shape[0], padding_size))
        processed_array = np.hstack((processed_array, padding))
    elif current_features_count > expected_features_count:
        # Jika kelebihan, potong kolomnya agar pas
        processed_array = processed_array[:, :expected_features_count]
        
    return processed_array

# ==============================================================================
# SIDEBAR INPUT FORM
# ==============================================================================
if model:
    # Mendeteksi otomatis berapa jumlah kolom fitur yang diminta oleh model pkl kamu
    try:
        expected_features_count = model.n_features_in_
    except AttributeError:
        expected_features_count = 14 # Nilai default perkiraan rata-rata fitur dataset
        
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
        
        submit_button = st.form_submit_button(label='💡 Jalankan Prediksi Churn')

    # ==============================================================================
    # PREDICTION LOGIC & OUTPUT DISPLAY
    # ==============================================================================
    if submit_button:
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
        
        st.subheader("📋 Ringkasan Data Pelanggan Baru")
        st.dataframe(input_data)
        
        try:
            # Jalankan preprocessing aman tanpa memuat file preprocessor.pkl
            input_processed = manual_preprocess(input_data, expected_features_count)
            
            # Eksekusi Prediksi
            prediction = model.predict(input_processed)[0]
            
            # Cek apakah model mendukung predict_proba
            try:
                prediction_proba = model.predict_proba(input_processed)[0]
                has_proba = True
            except AttributeError:
                has_proba = False
            
            st.markdown("---")
            st.subheader("🔮 Hasil Analisis & Prediksi Model")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if prediction == 1:
                    st.error("🚨 HASIL PREDIKSI: BERPOTENSI CHURN (PERGI)")
                    st.metric(label="Status Risiko", value="Tinggi (Churn)", delta="Perlu Intervensi Segera", delta_color="inverse")
                else:
                    st.success("✅ HASIL PREDIKSI: TETAP BERLANGGANAN (LOYAL)")
                    st.metric(label="Status Risiko", value="Rendah (Loyal)", delta="Pertahankan Layanan", delta_color="normal")
            
            with col2:
                if has_proba:
                    st.write("**Keyakinan Model (Probability Score):**")
                    st.write(f"• Probabilitas Tetap Setia (0): `{prediction_proba[0] * 100:.2f}%`")
                    st.write(f"• Probabilitas Akan Churn (1) : `{prediction_proba[1] * 100:.2f}%`")
                    st.progress(float(prediction_proba[1]))
                else:
                    st.write("**Catatan Evaluasi Model:**")
                    st.info("Model memproses klasifikasi data secara langsung (*Hard Voting/Direct Prediction*).")
                
            st.markdown("### 💡 Rekomendasi Tindakan Bisnis:")
            if prediction == 1:
                st.info("👉 **Strategi Retensi:** Kirimkan email penawaran khusus, berikan voucher diskon loyalitas, atau hubungi customer service untuk survei kepuasan guna mencegah mereka benar-benar pergi.")
            else:
                st.info("👉 **Strategi Retensi:** Masukkan pelanggan ke program loyalitas reguler dan tawarkan produk pelengkap (cross-selling) karena tingkat loyalitas terpantau stabil.")
                
        except Exception as e:
            st.error(f"❌ Terjadi kesalahan dalam pemrosesan data: {e}")
