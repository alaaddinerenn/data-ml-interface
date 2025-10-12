import streamlit as st

st.set_page_config(page_title="Data & ML Interface", page_icon="📊", layout='wide')

# Title at the top of the page (regardless of sidebar visibility)
st.markdown(
    """
    <h1 style="position: fixed; top: 40px; left: 0px; text-align: center;  font-size: 64px; width: 100%; margin: 0; z-index: 999;">
        Data & ML Interface
    </h1>
    """,
    unsafe_allow_html=True
)

# Add space so the page content starts lower
st.write("\n\n\n\n\n")
st.write("\n\n\n\n\n")

st.title("🏠 Home Page")
st.write("With this application, you can perform data analysis and run machine learning models.")
st.write("\n\n\n\n\n")
st.write("\n\n\n\n\n")

st.markdown(
    "<h2 style='text-align: center;'>📂 Navigate to Pages</h2>",
    unsafe_allow_html=True
)

# CSS
st.markdown(
    """
    <style>
    .card {
        display: flex;
        flex-direction: column; /* emoji on top, text below */
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
    .card-analysis { background-color: #056f2a; }
    .card-ml { background-color: #888282; }
    .card-about { background-color: #3498db; }

    .card:hover {
        transform: scale(1.05);
        box-shadow: 0px 4px 15px rgba(0,0,0,0.3);
        filter: brightness(1.1);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# 3-column layout
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<a class="card card-analysis" href="/Analysis" target="_self">📊<br>Analysis</a>', unsafe_allow_html=True)

with col2:
    st.markdown('<a class="card card-ml" href="/ML" target="_self">🤖<br>Machine Learning</a>', unsafe_allow_html=True)

with col3:
    st.markdown('<a class="card card-about" href="/About" target="_self">ℹ️<br>About</a>', unsafe_allow_html=True)
