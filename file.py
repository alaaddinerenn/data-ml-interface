from typing import Any
import streamlit as st
import pandas as pd

def load_file() -> None | Any:
    uploaded_file = st.file_uploader(
        "Upload a CSV or Excel file", 
        type=['csv', 'xlsx', 'tsv']
    )

    # If the file is removed (by clicking the cross)
    if uploaded_file is None:
        for key in ["df", "df_clean", "cleaned", "already_cleaned", "file_name"]:
            if key in st.session_state:
                del st.session_state[key]
        return None

    # If a new file is uploaded or it's the first upload
    if ('file_name' not in st.session_state) or (st.session_state.file_name != uploaded_file.name):
        st.session_state.file_name = uploaded_file.name
        st.session_state.cleaned = False  # Reset the cleaning flag
        
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(".tsv"):
                df = pd.read_csv(uploaded_file, sep="\t")
            else:
                df = pd.read_excel(uploaded_file)

            st.session_state.df = df  # Store the DataFrame
            st.success("File successfully uploaded and converted to a DataFrame!")
        except Exception as e:
            st.error(f"An error occurred while reading the file: {e}")

    return st.session_state.get("df", None)  # Always read from session_state
