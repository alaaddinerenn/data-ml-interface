import streamlit as st
from file import FileManager
from utils import DataComparator, DataCleaner
from stats import StatisticsDisplay

st.set_page_config(page_title="Analysis", page_icon="📊")

# Title
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

# Initialize session state
if 'cleaned' not in st.session_state:
    st.session_state.cleaned = False

# File upload
FileManager.load_file()

# Main analysis flow
if "df" in st.session_state and not st.session_state.df.empty:
    # Show initial statistics
    StatisticsDisplay.show_stats()

    # Data cleaning workflow
    if not st.session_state.cleaned:
        df_clean, st.session_state.cleaned, st.session_state.already_cleaned = DataCleaner.clean_data(
            st.session_state.df
        )
        st.session_state.df_clean = df_clean
        
        # If data is already clean, show button to send to ML
        if not st.session_state.already_cleaned:
            if st.button("📤 Send raw data to ML page", key="send_raw_before_clean"):
                st.session_state.df_for_ml_clean = None
                st.session_state.df_for_ml_raw = st.session_state.df
                st.switch_page("pages/02_ML.py")

    # After cleaning
    if st.session_state.cleaned:
        if not st.session_state.already_cleaned:
            # Show statistics after cleaning
            StatisticsDisplay.show_stats()
            
            # Compare before/after
            DataComparator.compare(st.session_state.df, st.session_state.df_clean)

            # Send data to ML page options
            st.markdown("---")
            st.subheader("📤 Send Data to ML Page")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("✅ Send cleaned data to ML page", key="send_clean", use_container_width=True):
                    st.session_state.df_for_ml_raw = None
                    st.session_state.df_for_ml_clean = st.session_state.df_clean
                    st.success("✅ Cleaned data prepared for ML!")
                    st.switch_page("pages/02_ML.py")
            
            with col2:
                if st.button("📄 Send raw data to ML page", key="send_raw", use_container_width=True):
                    st.session_state.df_for_ml_clean = None
                    st.session_state.df_for_ml_raw = st.session_state.df
                    st.success("✅ Raw data prepared for ML!")
                    st.switch_page("pages/02_ML.py")
        
        else:
            # Data was already clean
            st.markdown("---")
            if st.button("📤 Send data to ML page", key="send_any", use_container_width=True):
                st.session_state.df_for_ml_raw = st.session_state.df
                st.session_state.df_for_ml_clean = None
                st.success("✅ Data prepared for ML!")
                st.switch_page("pages/02_ML.py")
else:
    st.info("📂 Please upload a dataset to begin analysis.")