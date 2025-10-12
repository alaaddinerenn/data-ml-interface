import streamlit as st

st.set_page_config(page_title="Data & ML Interface", page_icon="ℹ️", layout='wide')


# Title at the top of the page (regardless of sidebar visibility)
st.markdown(
    """
    <h1 style="position: fixed; top: 40px; left: 20px; margin: 0; z-index: 999;">
        Data & ML Interface
    </h1>
    """,
    unsafe_allow_html=True
)

# Add space so the page content starts lower
st.write("\n\n\n\n\n")

st.title("ℹ️ About")
st.write("""
This application was developed using Streamlit.

### Features:
- 📂 Data upload (CSV & Excel)
- 🧹 Missing value cleaning
- 📊 Data analysis and visualization
- 🤖 Simple machine learning models

Developer: Alaaddin Eren Namlı
""")
