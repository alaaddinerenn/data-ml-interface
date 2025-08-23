import streamlit as st
import pandas as pd

def load_file():
    uploaded_file = st.file_uploader(
        "Bir CSV veya Excel dosyası yükleyin", 
        type=['csv', 'xlsx', 'tsv']
    )

    # Eğer dosya kaldırıldıysa (çarpıya basıldıysa)
    if uploaded_file is None:
        for key in ["df", "df_clean", "cleaned", "already_cleaned", "file_name"]:
            if key in st.session_state:
                del st.session_state[key]
        return None

    # Eğer yeni dosya yüklendiyse veya ilk defa yükleniyorsa
    if ('file_name' not in st.session_state) or (st.session_state.file_name != uploaded_file.name):
        st.session_state.file_name = uploaded_file.name
        st.session_state.cleaned = False  # temizleme flag'ini resetle
        
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(".tsv"):
                df = pd.read_csv(uploaded_file, sep="\t")
            else:
                df = pd.read_excel(uploaded_file)

            st.session_state.df = df  # DataFrame'i sakla
            st.success("Dosya başarıyla yüklendi ve DataFrame'e dönüştürüldü!")
        except Exception as e:
            st.error(f"Dosya okunurken hata oluştu: {e}")

    return st.session_state.get("df", None)  # Hep session_state'den oku
