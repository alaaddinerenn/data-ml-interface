import streamlit as st

st.set_page_config(page_title="Data & ML Arayüzü", page_icon="📊")

# Sayfa içeriğinin üstünde başlık (sidebar açık/kapalı fark etmez)
st.markdown(
    """
    <h1 style="position: fixed; top: 40px; left: 0px; text-align: center;  font-size: 64px; width: 100%; margin: 0; z-index: 999;">
        Data & ML Arayüzü
    </h1>
    """,
    unsafe_allow_html=True
)

# Sayfa içeriği biraz aşağıda başlasın diye boşluk bırak
st.write("\n\n\n\n\n")
st.write("\n\n\n\n\n")

st.title("🏠 Ana Sayfa")
st.write("Bu uygulama ile veri analizi yapabilir ve makine öğrenmesi modelleri çalıştırabilirsiniz.")
st.write("\n\n\n\n\n")
st.write("\n\n\n\n\n")

st.markdown(
    "<h2 style='text-align: center;'>📂 Sayfalara Git</h2>",
    unsafe_allow_html=True
)

# CSS
st.markdown(
    """
    <style>
    .card {
        display: flex;
        flex-direction: column; /* emoji üstte, text altta */
        align-items: center;
        justify-content: center;
        text-align: center;
        padding: 30px;
        border-radius: 10px;
        font-size: 24px;
        font-weight: bold;
        height: 150px;
        transition: all 0.3s ease;
        text-decoration: none !important;
        color: white !important;
        line-height: 1.4;
    }
    .card-analiz { background-color: #056f2a; }
    .card-ml { background-color: #888282; }
    .card-hakkinda { background-color: #3498db; }

    .card:hover {
        transform: scale(1.05);
        box-shadow: 0px 4px 15px rgba(0,0,0,0.3);
        filter: brightness(1.1);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# 3 kolonlu layout
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<a class="card card-analiz" href="/Analiz" target="_self">📊<br>Analiz</a>', unsafe_allow_html=True)

with col2:
    st.markdown('<a class="card card-ml" href="/ML" target="_self">🤖<br>Makine Öğrenmesi</a>', unsafe_allow_html=True)

with col3:
    st.markdown('<a class="card card-hakkinda" href="/Hakkında" target="_self">ℹ️<br>Hakkında</a>', unsafe_allow_html=True)