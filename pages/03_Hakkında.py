import streamlit as st

# Sayfa içeriğinin üstünde başlık (sidebar açık/kapalı fark etmez)
st.markdown(
    """
    <h1 style="position: fixed; top: 40px; left: 20px; margin: 0; z-index: 999;">
        Data&ML Arayüzü
    </h1>
    """,
    unsafe_allow_html=True
)

# Sayfa içeriği biraz aşağıda başlasın diye boşluk bırak
st.write("\n\n\n\n\n")

st.title("ℹ️ Hakkında")
st.write("""
Bu uygulama Streamlit kullanılarak geliştirilmiştir.

### Özellikler:
- 📂 Veri yükleme (CSV & Excel)
- 🧹 Eksik değer temizleme
- 📊 Veri analizi ve görselleştirme
- 🤖 Basit makine öğrenmesi modelleri

Geliştirici: Alaaddin Eren Namlı
""")
