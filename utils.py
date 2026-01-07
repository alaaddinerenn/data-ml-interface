import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import hashlib
from typing import Optional, Dict, List, Any
from sklearn.preprocessing import LabelEncoder


class DownloadManager:
    """Handles plot download functionality."""
    
    SUPPORTED_FORMATS = ["jpg", "png", "pdf"]
    MIME_TYPES = {
        "jpg": "image/jpeg",
        "png": "image/png",
        "pdf": "application/pdf"
    }
    
    @staticmethod
    def download_plot(
        fig,
        graph_type: str,
        feature_names: Optional[Any] = None,
        ext_default: str = "jpg"
    ) -> None:
        """
        Add download button for matplotlib figure.
        
        Args:
            fig: Matplotlib figure
            graph_type: Type of graph (e.g., 'histogram', 'scatter')
            feature_names: Feature name(s) used in plot
            ext_default: Default file extension
        """
        # Format feature names for filename
        if feature_names is None:
            feature_part = ""
        elif isinstance(feature_names, list):
            feature_part = "_vs_".join(str(f) for f in feature_names)
        else:
            feature_part = str(feature_names)
        
        feature_part = feature_part.replace(" ", "_")
        
        # Create unique hash-based keys to avoid duplicates
        # Combine graph_type and feature_part to create unique identifier
        unique_string = f"{graph_type}_{feature_part}_{id(fig)}"
        key_hash = hashlib.md5(unique_string.encode()).hexdigest()[:8]
        
        format_key = f"format_{key_hash}"
        dl_key = f"dl_{key_hash}"
        
        # Initialize format in session state
        if format_key not in st.session_state:
            st.session_state[format_key] = ext_default
        
        # Layout: format selector + download button
        col_empty1, col1, col2, col_empty2 = st.columns([1, 2, 2, 1])
        
        with col1:
            format_sec = st.selectbox(
                "Format",
                DownloadManager.SUPPORTED_FORMATS,
                key=format_key,
                index=DownloadManager.SUPPORTED_FORMATS.index(st.session_state[format_key]),
                label_visibility="collapsed"
            )
        
        # Create buffer and save figure
        buf = io.BytesIO()
        fig.savefig(buf, format=format_sec, bbox_inches="tight", dpi=300)
        buf.seek(0)
        
        filename = f"{graph_type}{'_' + feature_part if feature_part else ''}.{format_sec}"
        
        with col2:
            st.download_button(
                label="⬇️ Download",
                data=buf,
                file_name=filename,
                mime=DownloadManager.MIME_TYPES[format_sec],
                key=dl_key,
                use_container_width=True
            )


class DataComparator:
    """Compares datasets before and after cleaning."""
    
    @staticmethod
    def compare(df_before: pd.DataFrame, df_after: pd.DataFrame) -> None:
        """
        Display comprehensive comparison between two datasets.
        
        Args:
            df_before: Original dataset
            df_after: Cleaned dataset
        """
        st.markdown("### 🔍 Comparison: Before and After Data Cleaning")
        
        tab_num, tab_cat = st.tabs(["📊 Numerical Data", "📋 Categorical Data"])
        
        # Remove target columns
        target_cols = DataComparator._get_target_columns(df_before)
        dfb = df_before.drop(columns=target_cols, errors='ignore')
        dfa = df_after.drop(columns=target_cols, errors='ignore')
        
        # Separate numeric and categorical
        numeric_cols = DataComparator._get_numeric_columns(dfb)
        categorical_cols = [c for c in dfb.columns if c not in numeric_cols]
        
        with tab_num:
            DataComparator._compare_numerical(dfb, dfa, numeric_cols)
        
        with tab_cat:
            DataComparator._compare_categorical(dfb, dfa, categorical_cols)
    
    @staticmethod
    def _get_target_columns(df: pd.DataFrame) -> List[str]:
        """Identify target columns in dataset."""
        target_cols = []
        for col in ['label', 'target', 'cluster']:
            if col in df.columns:
                target_cols.append(col)
        target_cols.extend([c for c in df.columns if c.startswith('target')])
        return list(set(target_cols))
    
    @staticmethod
    def _get_numeric_columns(df: pd.DataFrame) -> List[str]:
        """Get list of numeric columns."""
        return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    
    @staticmethod
    def _compare_numerical(
        dfb: pd.DataFrame,
        dfa: pd.DataFrame,
        numeric_cols: List[str]
    ) -> None:
        """Compare numerical features."""
        if not numeric_cols:
            st.info("No numerical columns found.")
            return
        
        # Missing values
        st.markdown("#### 🧱 Missing Values")
        na_before = dfb[numeric_cols].isna().sum()
        na_after = dfa[numeric_cols].isna().sum()
        na_df = pd.DataFrame({"Before": na_before, "After": na_after})
        st.dataframe(na_df)
        st.bar_chart(na_df)
        
        # Statistical summary
        st.markdown("#### 📏 Statistical Summary")
        selected_cols = st.multiselect(
            "Select numerical columns",
            numeric_cols,
            default=numeric_cols[:min(5, len(numeric_cols))],
            key="compare_num_cols"
        )
        
        if selected_cols:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Before Cleaning**")
                st.dataframe(dfb[selected_cols].describe().round(3))
            with col2:
                st.markdown("**After Cleaning**")
                st.dataframe(dfa[selected_cols].describe().round(3))
        
        # Histogram comparison
        st.markdown("#### 📉 Distribution Comparison")
        hist_cols = st.multiselect(
            "Select features for histogram",
            numeric_cols,
            default=numeric_cols[:min(3, len(numeric_cols))],
            key="compare_hist_cols"
        )
        
        for col in hist_cols:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            
            sns.histplot(dfb[col].dropna(), ax=axes[0], color='orange', kde=True, bins=20)
            axes[0].set_title(f"Before: {col}")
            axes[0].grid(True, alpha=0.3)
            
            sns.histplot(dfa[col].dropna(), ax=axes[1], color='blue', kde=True, bins=20)
            axes[1].set_title(f"After: {col}")
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
    
    @staticmethod
    def _compare_categorical(
        dfb: pd.DataFrame,
        dfa: pd.DataFrame,
        categorical_cols: List[str]
    ) -> None:
        """Compare categorical features."""
        if not categorical_cols:
            st.info("No categorical columns found.")
            return
        
        # Missing values
        st.markdown("#### 🧱 Missing Values")
        na_before = dfb[categorical_cols].isna().sum()
        na_after = dfa[categorical_cols].isna().sum()
        na_df = pd.DataFrame({"Before": na_before, "After": na_after})
        st.dataframe(na_df)
        st.bar_chart(na_df)
        
        # Detailed comparison
        selected_cols = st.multiselect(
            "Select categorical columns",
            categorical_cols,
            default=categorical_cols[:min(3, len(categorical_cols))],
            key="compare_cat_cols"
        )
        
        for col in selected_cols:
            DataComparator._compare_single_categorical(dfb, dfa, col)
    
    @staticmethod
    def _compare_single_categorical(
        dfb: pd.DataFrame,
        dfa: pd.DataFrame,
        col: str
    ) -> None:
        """Compare single categorical column in detail."""
        st.markdown(f"#### 📊 {col}")
        
        # Unique value count
        unique_before = dfb[col].nunique(dropna=False)
        unique_after = dfa[col].nunique(dropna=False)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Unique Values (Before)", unique_before)
        with col2:
            st.metric("Unique Values (After)", unique_after)
        
        # Get top values from BOTH before and after, then combine
        top_n = 10
        
        # Get top values from both datasets
        before_counts = dfb[col].fillna("NaN").value_counts()
        after_counts = dfa[col].fillna("NaN").value_counts()
        
        # Get top N from each
        before_top_values = set(before_counts.head(top_n).index)
        after_top_values = set(after_counts.head(top_n).index)
        
        # Combine and get union of top values from both
        all_top_values = sorted(before_top_values | after_top_values)
        
        # Limit to top_n most frequent overall (based on combined frequency)
        combined_counts = {}
        for val in all_top_values:
            combined_counts[val] = before_counts.get(val, 0) + after_counts.get(val, 0)
        
        # Sort by combined frequency and take top_n
        top_values = sorted(
            combined_counts.keys(), 
            key=lambda x: combined_counts[x], 
            reverse=True
        )[:top_n]
        
        # Create frequency DataFrame
        freq_df = pd.DataFrame({
            "Before": [before_counts.get(val, 0) for val in top_values],
            "After": [after_counts.get(val, 0) for val in top_values]
        }, index=top_values)
        
        # Add "Others" category (values not in top_n)
        others_before = before_counts.drop(
            index=top_values, errors='ignore'
        ).sum()
        others_after = after_counts.drop(
            index=top_values, errors='ignore'
        ).sum()
        
        # Only add "Others" if there are other values
        if others_before > 0 or others_after > 0:
            freq_df.loc["Others"] = [others_before, others_after]
        
        # Display table
        st.dataframe(freq_df)
        
        # Bar chart
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create bar positions
        x = range(len(freq_df))
        width = 0.35
        
        # Plot bars
        ax.bar(
            [i - width/2 for i in x], 
            freq_df["Before"], 
            width, 
            label='Before', 
            color='orange',
            alpha=0.8
        )
        ax.bar(
            [i + width/2 for i in x], 
            freq_df["After"], 
            width, 
            label='After', 
            color='blue',
            alpha=0.8
        )
        
        # Customize plot
        ax.set_xlabel("Category", fontweight='bold')
        ax.set_ylabel("Count", fontweight='bold')
        ax.set_title(f"Frequency Comparison: {col}", fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(freq_df.index, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


class DataCleaner:
    """Handles data cleaning operations."""
    
    MISSING_VALUE_INDICATORS = [
        "None", "NA", "Missing", "?", "", "na", "NaN", "N/A", "n/a"
    ]
    
    @staticmethod
    def clean_data(df: pd.DataFrame) -> tuple:
        """
        Interactive data cleaning interface.
        
        Args:
            df: DataFrame to clean
            
        Returns:
            Tuple of (cleaned_df, is_cleaned, was_already_clean)
        """
        if df is None or df.empty:
            st.warning("⚠️ No data available.")
            return df, False, False
        
        # Replace missing value indicators
        df_cleaned = df.copy()
        df_cleaned.replace(DataCleaner.MISSING_VALUE_INDICATORS, pd.NA, inplace=True)
        
        # Find columns with missing values
        na_cols = df_cleaned.columns[df_cleaned.isna().any()].tolist()
        
        if not na_cols:
            st.info("✅ No missing values found. Data is already clean!")
            return df_cleaned, True, True
        
        st.markdown("### 🧹 Missing Value Treatment")
        st.info(f"Found {len(na_cols)} columns with missing values.")
        
        actions = DataCleaner._get_cleaning_actions(df_cleaned, na_cols)
        
        if st.button("🧹 Apply Cleaning", type="primary"):
            df_cleaned = DataCleaner._apply_cleaning(df_cleaned, actions)
            st.success("✅ Data cleaning completed!")
            return df_cleaned, True, False
        
        return df, False, False
    
    @staticmethod
    def _get_cleaning_actions(df: pd.DataFrame, na_cols: List[str]) -> Dict[str, str]:
        """Get cleaning action for each column."""
        actions = {}
        
        for col in na_cols:
            missing_count = df[col].isna().sum()
            missing_pct = (missing_count / len(df)) * 100
            
            st.markdown(f"**{col}**")
            st.caption(f"Missing: {missing_count} rows ({missing_pct:.2f}%)")
            
            is_numeric = pd.api.types.is_numeric_dtype(df[col])
            
            if is_numeric:
                options = [
                    "Do nothing",
                    "Fill with mean",
                    "Fill with median",
                    "Fill with mode",
                    "Drop rows"
                ]
            else:
                options = [
                    "Do nothing",
                    "Fill with mode",
                    "Fill with custom value",
                    "Drop rows"
                ]
            
            action = st.selectbox(
                f"Action for {col}",
                options,
                key=f"na_action_{col}",
                label_visibility="collapsed"
            )
            
            actions[col] = action
        
        return actions
    
    @staticmethod
    def _apply_cleaning(df: pd.DataFrame, actions: Dict[str, str]) -> pd.DataFrame:
        """Apply cleaning actions to DataFrame."""
        for col, action in actions.items():
            if action == "Fill with mean":
                df[col] = df[col].fillna(df[col].mean())
            
            elif action == "Fill with median":
                df[col] = df[col].fillna(df[col].median())
            
            elif action == "Fill with mode":
                mode_val = df[col].mode()
                if not mode_val.empty:
                    df[col] = df[col].fillna(mode_val[0])
            
            elif action == "Fill with custom value":
                # This would need additional input - simplified for now
                df[col] = df[col].fillna("Unknown")
            
            elif action == "Drop rows":
                df = df[df[col].notna()]
        
        return df