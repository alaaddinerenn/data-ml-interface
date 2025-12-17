import streamlit as st
import pandas as pd
from typing import Optional
import io


class FileManager:
    """Handles file upload and dataset management."""
    
    SUPPORTED_EXTENSIONS = ['csv', 'xlsx', 'xls', 'tsv', 'json', 'parquet']
    
    @staticmethod
    def load_file() -> Optional[pd.DataFrame]:
        """
        Display file uploader and handle file loading.
        
        Returns:
            DataFrame if file is loaded, None otherwise
        """
        uploaded_file = st.file_uploader(
            "📁 Upload your dataset",
            type=FileManager.SUPPORTED_EXTENSIONS,
            help="Supported formats: CSV, Excel, TSV, JSON, Parquet"
        )
        
        # File removed - clear session state
        if uploaded_file is None:
            FileManager._clear_session_state()
            return None
        
        # New file or first upload
        if FileManager._is_new_file(uploaded_file):
            df = FileManager._process_file(uploaded_file)
            if df is not None:
                FileManager._store_in_session(uploaded_file.name, df)
            return df
        
        # Return existing DataFrame from session
        return st.session_state.get("df", None)
    
    @staticmethod
    def _is_new_file(uploaded_file) -> bool:
        """Check if uploaded file is new."""
        return ('file_name' not in st.session_state or 
                st.session_state.file_name != uploaded_file.name)
    
    @staticmethod
    def _clear_session_state() -> None:
        """Clear file-related session state."""
        keys_to_clear = [
            "df", "df_clean", "cleaned", "already_cleaned",
            "file_name", "df_for_ml_clean", "df_for_ml_raw"
        ]
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
    
    @staticmethod
    def _process_file(uploaded_file) -> Optional[pd.DataFrame]:
        """
        Process uploaded file and convert to DataFrame.
        
        Args:
            uploaded_file: Streamlit UploadedFile object
            
        Returns:
            DataFrame if successful, None otherwise
        """
        try:
            filename = uploaded_file.name.lower()
            
            if filename.endswith('.csv'):
                df = FileManager._read_csv(uploaded_file)
            elif filename.endswith('.tsv'):
                df = pd.read_csv(uploaded_file, sep='\t')
            elif filename.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(uploaded_file)
            elif filename.endswith('.json'):
                df = pd.read_json(uploaded_file)
            elif filename.endswith('.parquet'):
                df = pd.read_parquet(uploaded_file)
            else:
                st.error(f"❌ Unsupported file format: {filename}")
                return None
            
            # Validate DataFrame
            if df.empty:
                st.warning("⚠️ The uploaded file is empty.")
                return None
            
            # Show success message with info
            st.success(f"✅ File '{uploaded_file.name}' loaded successfully!")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", df.shape[0])
            with col2:
                st.metric("Columns", df.shape[1])
            with col3:
                memory_usage = df.memory_usage(deep=True).sum() / 1024**2
                st.metric("Size", f"{memory_usage:.2f} MB")
            
            return df
            
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
            st.exception(e)
            return None
    
    @staticmethod
    def _read_csv(uploaded_file) -> pd.DataFrame:
        """
        Read CSV file with automatic delimiter detection.
        
        Args:
            uploaded_file: Streamlit UploadedFile object
            
        Returns:
            DataFrame
        """
        # Try common delimiters
        delimiters = [',', ';', '\t', '|']
        
        # Read first few lines to detect delimiter
        content = uploaded_file.read().decode('utf-8')
        uploaded_file.seek(0)  # Reset file pointer
        
        for delimiter in delimiters:
            if delimiter in content[:1000]:  # Check first 1000 chars
                try:
                    df = pd.read_csv(io.StringIO(content), sep=delimiter)
                    if df.shape[1] > 1:  # Valid separation
                        return df
                except:
                    continue
        
        # Fallback to default
        return pd.read_csv(uploaded_file)
    
    @staticmethod
    def _store_in_session(filename: str, df: pd.DataFrame) -> None:
        """
        Store DataFrame in session state.
        
        Args:
            filename: Name of uploaded file
            df: DataFrame to store
        """
        st.session_state.file_name = filename
        st.session_state.df = df
        st.session_state.cleaned = False
        
        # Clear previous ML data
        if 'df_for_ml_clean' in st.session_state:
            del st.session_state.df_for_ml_clean
        if 'df_for_ml_raw' in st.session_state:
            del st.session_state.df_for_ml_raw