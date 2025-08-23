import streamlit as st
from file import load_file
from utils import compare, clean_data
from stats import show_stats

st.markdown(
    """
    <h1 style="position: fixed; top: 40px; left: 20px; margin: 0; z-index: 999;">
        Data&ML Arayüzü
    </h1>
    """,
    unsafe_allow_html=True
)

st.write("\n\n\n\n\n")
st.title("🔍 Analiz Sayfası")

if 'cleaned' not in st.session_state:
    st.session_state.cleaned = False

# Dosya yükleme
load_file()  # df döndürmüyor artık, direkt session_state.df içine yazıyor

if "df" in st.session_state and not st.session_state.df.empty:
    show_stats()

    if not st.session_state.cleaned:
        df_clean, st.session_state.cleaned , st.session_state.already_cleaned = clean_data(st.session_state.df)
        st.session_state.df_clean = df_clean
        if not st.session_state.already_cleaned and st.button("Ham veriyi ML sayfasına gönder"): # temizlemeden gönder.
                    st.session_state.df_for_ml_clean = None
                    st.session_state.df_for_ml_raw = st.session_state.df
                    st.switch_page("pages/02_ML.py")

    if st.session_state.cleaned:
        if not st.session_state.already_cleaned:
            show_stats()
            compare(st.session_state.df, st.session_state.df_clean)

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Temizlenmiş veriyi ML sayfasına gönder", key="send_clean"):
                    st.session_state.df_for_ml_raw = None
                    st.session_state.df_for_ml_clean = st.session_state.df_clean
                    st.switch_page("pages/02_ML.py")
            with col2:
                if st.button("Ham veriyi ML sayfasına gönder", key="send_raw"):
                    st.session_state.df_for_ml_clean = None
                    st.session_state.df_for_ml_raw = st.session_state.df
                    st.switch_page("pages/02_ML.py")
        else:
            if st.button("Veriyi ML sayfasına gönder", key="send_any"):
                st.session_state.df_for_ml_raw = st.session_state.df
                st.session_state.df_for_ml_clean = None
                st.switch_page("pages/02_ML.py")
else:
    st.info("Lütfen bir veri seti yükleyin.")
