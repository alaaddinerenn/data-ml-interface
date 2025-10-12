import streamlit as st
from file import load_file
from utils import compare, clean_data
from stats import show_stats

st.set_page_config(page_title="Analysis", page_icon="📊", layout='wide')

st.markdown(
    """
    <h1 style="text-align: center; font-size: 50px; margin: 10px 0 30px 0;">
        Data & ML Interface
    </h1>
    """,
    unsafe_allow_html=True
)

st.write("\n\n\n\n\n")
st.title("🔍 Analysis Page")

if 'cleaned' not in st.session_state:
    st.session_state.cleaned = False

# File upload
load_file()  # No longer returns df, directly writes to session_state.df

if "df" in st.session_state and not st.session_state.df.empty:
    show_stats()

    if not st.session_state.cleaned:
        df_clean, st.session_state.cleaned , st.session_state.already_cleaned = clean_data(st.session_state.df)
        st.session_state.df_clean = df_clean
        if not st.session_state.already_cleaned and st.button("Send raw data to ML page"):
                    st.session_state.df_for_ml_clean = None
                    st.session_state.df_for_ml_raw = st.session_state.df
                    st.switch_page("pages/02_ML.py")

    if st.session_state.cleaned:
        if not st.session_state.already_cleaned:
            show_stats()
            compare(st.session_state.df, st.session_state.df_clean)

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Send cleaned data to ML page", key="send_clean"):
                    st.session_state.df_for_ml_raw = None
                    st.session_state.df_for_ml_clean = st.session_state.df_clean
                    st.switch_page("pages/02_ML.py")
            with col2:
                if st.button("Send raw data to ML page", key="send_raw"):
                    st.session_state.df_for_ml_clean = None
                    st.session_state.df_for_ml_raw = st.session_state.df
                    st.switch_page("pages/02_ML.py")
        else:
            if st.button("Send data to ML page", key="send_any"):
                st.session_state.df_for_ml_raw = st.session_state.df
                st.session_state.df_for_ml_clean = None
                st.switch_page("pages/02_ML.py")
else:
    st.info("Please upload a dataset.")
