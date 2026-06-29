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
        gender = st.selectbox(
            "Gender",
            ["Female", "Male"]
        )
        age = st.number_input(
            "Usia",
            min_value=18,
            max_value=100,
            value=30
        )
        country = st.selectbox(
            "Country",
            [
                "Bangladesh",
                "Germany",
                "India",
                "UK",
                "USA"
            ]
        )
        city = st.selectbox(
            "Kota",
            [
                "Berlin",
                "Delhi",
                "Dhaka",
                "Hamburg",
                "London",
                "Mumbai",
                "New York"
            ]
        )
        acquisition_channel = st.selectbox(
            "Acquisition Channel",
            [
                "Email",
                "Facebook Ads",
                "Google Ads",
                "Organic",
                "Referral"
            ]
        )
        device_type = st.selectbox(
            "Device Type",
            [
                "Desktop",
                "Mobile",
                "Tablet"
            ]
        )
        subscription_type = st.selectbox(
            "Subscription Type",
            [
                "Annual",
                "Monthly"
            ]
        )
        is_premium_user = st.selectbox(
            "Premium User",
            [0, 1]
        )        
        
        st.markdown("---")
        st.subheader("📈 Aktivitas & Transaksi")
        total_visits = st.number_input(
            "Total Visits",
            min_value=0,
            value=50
        )
        avg_session_time = st.number_input(
            "Average Session Time",
            min_value=0.0,
            value=30.0
        )
        pages_per_session = st.number_input(
            "Pages Per Session",
            min_value=0.0,
            value=5.0
        )
        email_open_rate = st.slider(
            "Email Open Rate",
            0.0,
            1.0,
            0.5
        )
        email_click_rate = st.slider(
            "Email Click Rate",
            0.0,
            1.0,
            0.2
        )
        total_spent = st.number_input("Total Spent (Total nominal pengeluaran)", min_value=0.0, value=500.0)
        avg_order_value = st.number_input(
            "Average Order Value",
            min_value=0.0,
            value=100.0
        )
        discount_used = st.number_input(
            "Discount Used",
            min_value=0,
            value=2
        )
        coupon_code = st.selectbox(
            "Coupon Code",
            [
                "NEW20",
                "REF10",
                "SALE15"
            ]
        )
        support_tickets = st.number_input(
            "Support Tickets",
            min_value=0,
            value=1
        )
        refund_requested = st.selectbox(
            "Refund Requested",
            [0, 1]
        )
        delivery_delay_days = st.number_input("Rata-rata Keterlambatan Pengiriman (Hari)", min_value=0, max_value=30, value=2)
        payment_method = st.selectbox(
            "Payment Method",
            [
                "BKash",
                "Card",
                "PayPal",
                "SEPA",
                "UPI"
            ]
        )
        satisfaction_score = st.slider("Skor Kepuasan Pelanggan (1-5)", min_value=1, max_value=5, value=4)
        nps_score = st.slider(
            "NPS Score",
            -100,
            100,
            20
        )
        marketing_spend_per_user = st.number_input(
            "Marketing Spend Per User",
            min_value=0.0,
            value=100.0
        )
        lifetime_value = st.number_input("Lifetime Value (CLV)", min_value=0.0, value=1500.0)
        last_3_month_purchase_freq = st.number_input(
            "Last 3 Month Purchase Frequency",
            min_value=0,
            value=5
        )
        
        # Tombol Submit Form
        submit_button = st.form_submit_button(label='Jalankan Prediksi Churn')

    # ==============================================================================
    # PREDICTION LOGIC & OUTPUT DISPLAY
    # ==============================================================================
    if submit_button:
        # 1. Mengubah input user menjadi DataFrame (sesuai nama kolom dataset asli)
        input_data = pd.DataFrame([{
            "gender": gender,
            "age": age,
            "country": country,
            "city": city,
            "acquisition_channel": acquisition_channel,
            "device_type": device_type,
            "subscription_type": subscription_type,
            "is_premium_user": is_premium_user,
            "total_visits": total_visits,
            "avg_session_time": avg_session_time,
            "pages_per_session": pages_per_session,
            "email_open_rate": email_open_rate,
            "email_click_rate": email_click_rate,
            "total_spent": total_spent,
            "avg_order_value": avg_order_value,
            "discount_used": discount_used,
            "coupon_code": coupon_code,
            "support_tickets": support_tickets,
            "refund_requested": refund_requested,
            "delivery_delay_days": delivery_delay_days,
            "payment_method": payment_method,
            "satisfaction_score": satisfaction_score,
            "nps_score": nps_score,
            "marketing_spend_per_user": marketing_spend_per_user,
            "lifetime_value": lifetime_value,
            "last_3_month_purchase_freq": last_3_month_purchase_freq
        }])
        
        # Menampilkan Ringkasan Data Input di Area Utama
        st.subheader("📋 Ringkasan Data Pelanggan Baru")
        st.dataframe(input_data)
        
        try:
            # 2. Transformasi data menggunakan preprocessor (Scaling & Encoding)
            input_processed = preprocessor.transform(input_data)

            # ==========================
            # DEBUG
            # ==========================
            st.write("### Debug - Shape hasil preprocessing")
            st.write(input_processed.shape)

            # 3. Prediksi menggunakan model
            prediction = model.predict(input_processed)[0]
            prediction_proba = model.predict_proba(input_processed)[0]

            st.write("### Debug - Raw Probability")
            st.write(prediction_proba)

            st.markdown("---")
            st.subheader("🔮 Hasil Analisis & Prediksi Model")

            # Membagi layout visual menjadi 2 kolom
            col1, col2 = st.columns(2)

            with col1:
                if prediction == 1:
                    st.error("🚨 HASIL PREDIKSI: BERPOTENSI CHURN (PERGI)")
                    st.metric(
                        label="Status Risiko",
                        value="Tinggi (Churn)",
                        delta="Perlu Intervensi Segera",
                        delta_color="inverse"
                    )
                else:
                    st.success("✅ HASIL PREDIKSI: TETAP BERLANGGANAN (LOYAL)")
                    st.metric(
                        label="Status Risiko",
                        value="Rendah (Loyal)",
                        delta="Pertahankan Layanan",
                        delta_color="normal"
                    )

            with col2:
                st.write("**Keyakinan Model (Probability Score):**")
                st.write(f"• Probabilitas Tetap Setia (0): `{prediction_proba[0] * 100:.2f}%`")
                st.write(f"• Probabilitas Akan Churn (1): `{prediction_proba[1] * 100:.2f}%`")

                # Visualisasi tingkat risiko churn
                st.progress(float(prediction_proba[1]))

            # Rekomendasi Bisnis
            st.markdown("### 💡 Rekomendasi Tindakan Bisnis:")

            if prediction == 1:
                st.info(
                    "👉 **Strategi Retensi:** Kirimkan email penawaran khusus, "
                    "berikan voucher diskon loyalitas, atau hubungi customer service "
                    "untuk survei kepuasan guna mencegah pelanggan benar-benar pergi."
                )
            else:
                st.info(
                    "👉 **Strategi Retensi:** Masukkan pelanggan ke program loyalitas "
                    "reguler dan tawarkan produk pelengkap (cross-selling/upselling) "
                    "karena retensi mereka terpantau stabil."
                )

        except Exception as e:
            st.error(f"❌ Terjadi kesalahan dalam pemrosesan data: {e}")
            st.warning(
                "Catatan: Pastikan urutan dan nama kolom input sama persis dengan "
                "fitur yang digunakan saat melatih model."
            )