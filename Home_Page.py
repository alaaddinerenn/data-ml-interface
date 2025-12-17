import streamlit as st

st.set_page_config(
    page_title="Data & ML Interface",
    page_icon="📊",
    layout='wide',
    initial_sidebar_state="collapsed"
)

# Fixed title at the top
st.markdown(
    """
    <h1 style="position: fixed; top: 40px; left: 0px; text-align: center; 
                font-size: 64px; width: 100%; margin: 0; z-index: 999;">
        Data & ML Interface
    </h1>
    """,
    unsafe_allow_html=True
)

# Add spacing
st.write("\n" * 10)

# Welcome section
st.title("🏠 Home Page")
st.write(
    "Welcome to **Data & ML Interface** - Your comprehensive tool for "
    "data analysis and machine learning."
)

st.write("\n" * 5)

# Navigation section
st.markdown(
    "<h2 style='text-align: center;'>📂 Navigate to Pages</h2>",
    unsafe_allow_html=True
)

# Enhanced CSS with smooth animations
st.markdown(
    """
    <style>
    .card {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        padding: 30px;
        border-radius: 15px;
        font-size: 24px;
        font-weight: bold;
        height: 150px;
        transition: all 0.3s ease;
        text-decoration: none !important;
        color: white !important;
        line-height: 1.4;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        cursor: pointer;
    }
    
    .card-analysis { 
        background: linear-gradient(135deg, #056f2a 0%, #0a9e3f 100%);
    }
    
    .card-ml { 
        background: linear-gradient(135deg, #888282 0%, #5a5656 100%);
    }
    
    .card-about { 
        background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
    }

    .card:hover {
        transform: translateY(-10px) scale(1.05);
        box-shadow: 0 12px 24px rgba(0,0,0,0.3);
        filter: brightness(1.15);
    }
    
    .emoji {
        font-size: 48px;
        display: block;
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# 3-column layout with enhanced cards
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        '<a class="card card-analysis" href="/Analysis" target="_self">'
        '<span class="emoji">📊</span>Analysis</a>',
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        '<a class="card card-ml" href="/ML" target="_self">'
        '<span class="emoji">🤖</span>Machine Learning</a>',
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        '<a class="card card-about" href="/About" target="_self">'
        '<span class="emoji">ℹ️</span>About</a>',
        unsafe_allow_html=True
    )

# Footer
st.write("\n" * 5)
st.markdown(
    """
    <div style='text-align: center; color: #888; font-size: 14px;'>
        <p>💡 Tip: Start with <strong>Analysis</strong> to explore your data, 
        then proceed to <strong>Machine Learning</strong> for modeling.</p>
    </div>
    """,
    unsafe_allow_html=True
)
