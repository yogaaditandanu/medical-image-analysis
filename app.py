# =====================
# IMPORT
# =====================
import os
import time
from dotenv import load_dotenv
import streamlit as st
from PIL import Image as PILImage
from agno.agent import Agent
from agno.models.google import Gemini
from agno.media import Image as AgnoImage

# =====================
# LOAD ENV & CONFIG
# =====================
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("❌ GOOGLE_API_KEY tidak ditemukan di file .env")
    st.stop()

# =====================
# AI AGENT (Update ke Model 2026)
# =====================
medical_agent = Agent(
    # 'gemini-2.5-flash' adalah standar stabil terbaru untuk kecepatan & visi
    # Kamu juga bisa mencoba 'gemini-3-flash' untuk teknologi paling mutakhir
    model=Gemini(id="gemini-2.5-flash"), 
    tools=[],
    markdown=True
)

# =====================
# BASE PROMPT
# =====================
BASE_QUERY = """
Anda adalah asisten AI analisis citra medis yang bertugas secara bertanggung jawab dan etis.

1. Validasi Gambar: Tentukan apakah gambar merupakan citra medis valid (X-ray, CT, MRI, USG). Jika bukan, nyatakan tidak dapat dianalisis.
2. Jenis & Area Gambar: Identifikasi jenis pencitraan dan area anatomi.
3. Temuan Utama: Jelaskan kelainan atau temuan secara sistematis.
4. Penilaian Diagnostik: Diagnosis paling mungkin dan 3 diagnosis banding.
5. Risiko: Skor keyakinan (0.0-1.0) dan Klasifikasi risiko (Rendah/Sedang/Tinggi).
6. Ringkasan: Maksimal 5 poin eksekutif.
7. Rekomendasi: Saran umum tanpa dosis obat.
8. Penjelasan Ramah Pasien: Gunakan bahasa yang mudah dimengerti.

Analisis ini hanya untuk edukasi, bukan diagnosis resmi.
"""

# =====================
# STREAMLIT UI SETUP
# =====================
st.set_page_config(page_title="Analisis Citra Medis", layout="centered")
st.title("🩺 Analisis Citra Medis Berbasis AI")

st.warning(
    "⚠️ Alat ini hanya untuk tujuan edukasi dan tidak menggantikan diagnosis tenaga medis profesional."
)

# =====================
# SESSION STATE INITIALIZATION
# =====================
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "last_file_name" not in st.session_state:
    st.session_state.last_file_name = None

# =====================
# INPUT MODE & UPLOAD
# =====================
mode = st.radio("Pilih mode penjelasan:", ["Tenaga Medis", "Pasien / Umum"])
query = BASE_QUERY + ("\nGunakan istilah medis profesional." if mode == "Tenaga Medis" else "\nGunakan bahasa yang sangat sederhana.")

uploaded_file = st.file_uploader(
    "Unggah gambar medis (X-ray, CT, MRI, USG)", 
    type=["jpg", "jpeg", "png"]
)

# =====================
# LOGIKA PROSES & VALIDASI
# =====================
if uploaded_file is not None:
    # 1. Reset hasil jika ganti file
    if uploaded_file.name != st.session_state.last_file_name:
        st.session_state.analysis_result = None
        st.session_state.last_file_name = uploaded_file.name

    # 2. Tampilkan Gambar
    st.image(uploaded_file, caption="Gambar yang diunggah", use_container_width=True)
    
    # 3. Validasi Nama File (Opsional - Hanya Peringatan)
    filename = uploaded_file.name.lower()
    keywords = ["xray", "ct", "mri", "usg", "ultrasound", "scan", "radiology"]
    if not any(k in filename for k in keywords):
        st.info("ℹ️ Tips: Pastikan gambar yang diunggah benar-benar citra medis untuk hasil akurat.")

    # 4. Tombol Analisis
    if st.button("🔍 Mulai Analisis Gambar"):
        with st.spinner("🧠 AI sedang menganalisis... Mohon tunggu."):
            try:
                # Simpan sementara untuk Agno
                temp_path = "temp_medical_image.png"
                img = PILImage.open(uploaded_file)
                img.save(temp_path)
                
                agno_img = AgnoImage(filepath=temp_path)
                
                # Eksekusi Agent
                response = medical_agent.run(query, images=[agno_img])
                
                # Cek konten respon untuk error tersembunyi
                res_text = str(response.content)
                if "429" in res_text:
                    st.error("🚨 Batas permintaan tercapai (Rate Limit). Tunggu 60 detik.")
                elif "404" in res_text:
                    st.error("🚨 Model AI tidak ditemukan (404). Coba ganti Model ID.")
                else:
                    st.session_state.analysis_result = res_text
                
                # Bersihkan file temp
                if os.path.exists(temp_path):
                    os.remove(temp_path)

            except Exception as e:
                error_str = str(e)
                if "429" in error_str:
                    st.error("🚨 Terlalu banyak permintaan! Google membatasi akses gratis. Coba lagi dalam 1 menit.")
                elif "404" in error_str:
                    st.error("❌ Model API tidak merespon (404). Pastikan library 'agno' terbaru.")
                else:
                    st.error(f"❌ Terjadi kesalahan teknis: {error_str}")

# =====================
# TAMPILKAN HASIL
# =====================
if st.session_state.analysis_result:
    st.divider()
    st.subheader("📋 Hasil Analisis")
    st.markdown(st.session_state.analysis_result)
    
    # Tombol reset manual
    if st.button("Clear Results"):
        st.session_state.analysis_result = None
        st.rerun()
else:
    if uploaded_file is None:
        st.info("Silakan unggah file gambar untuk memulai.")