import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Optional, List
from plotting import PlottingTools
from utils import DownloadManager


class StatisticsDisplay:
    """Handles statistical analysis and visualization display."""
    
    @staticmethod
    def show_stats(df: Optional[pd.DataFrame] = None) -> None:
        """
        Display comprehensive statistics for dataset.
        
        Args:
            df: DataFrame to analyze (uses session state if None)
        """
        # Get DataFrame from session state if not provided
        if df is None:
            df = StatisticsDisplay._get_dataframe_from_session()
        
        if df is None or df.empty:
            st.warning("⚠️ No data available for analysis.")
            return
        
        # Display basic info
        StatisticsDisplay._show_basic_info(df)
        
        # Determine data type and show appropriate analysis
        data_type, target_cols = StatisticsDisplay._determine_data_type(df)
        
        if data_type == "classification":
            StatisticsDisplay._show_classification_analysis(df, target_cols)
        elif data_type == "clustering":
            StatisticsDisplay._show_clustering_analysis(df, target_cols)
        elif data_type == "regression":
            StatisticsDisplay._show_regression_analysis(df, target_cols)
        else:
            StatisticsDisplay._show_generic_analysis(df)
    
    @staticmethod
    def _get_dataframe_from_session() -> Optional[pd.DataFrame]:
        """Retrieve DataFrame from session state."""
        if "df_clean" in st.session_state and not st.session_state.df_clean.empty:
            return st.session_state.df_clean
        elif "df" in st.session_state and not st.session_state.df.empty:
            return st.session_state.df
        return None
    
    @staticmethod
    def _show_basic_info(df: pd.DataFrame) -> None:
        """Display basic dataset information."""
        st.markdown("### 📊 Dataset Overview")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows", df.shape[0])
        with col2:
            st.metric("Columns", df.shape[1])
        with col3:
            st.metric("Missing Values", df.isna().sum().sum())
        
        # Show data
        with st.expander("View Data", expanded=False):
            st.dataframe(df)
        
        # Column info
        with st.expander("Column Information"):
            col_info = pd.DataFrame({
                "Column": df.columns,
                "Data Type": df.dtypes.astype(str),
                "Non-Null Count": df.count(),
                "Null Count": df.isna().sum(),
                "Unique Values": df.nunique()
            })
            st.dataframe(col_info)
    
    @staticmethod
    def _determine_data_type(df: pd.DataFrame) -> tuple:
        """
        Determine the type of dataset.
        
        Returns:
            Tuple of (data_type, target_columns)
        """
        if 'label' in df.columns:
            return "classification", ['label']
        elif 'cluster' in df.columns:
            return "clustering", ['cluster']
        elif 'target' in df.columns:
            return "regression", ['target']
        elif any(col.startswith("target") for col in df.columns):
            target_cols = [col for col in df.columns if col.startswith("target")]
            return "multi_regression", target_cols
        
        return "generic", []
    
    @staticmethod
    def _show_classification_analysis(df: pd.DataFrame, target_cols: List[str]) -> None:
        """Show analysis for classification data."""
        st.markdown("### 🎯 Classification Analysis")
        
        target_col = target_cols[0]
        
        # Class distribution
        StatisticsDisplay._show_class_distribution(df, target_col)
        
        # Feature analysis
        feature_df = df.drop(columns=target_cols)
        StatisticsDisplay._show_statistical_summary(feature_df)
        
        # Visualizations
        StatisticsDisplay._show_feature_visualizations(df, target_col)
    
    @staticmethod
    def _show_clustering_analysis(df: pd.DataFrame, target_cols: List[str]) -> None:
        """Show analysis for clustering data."""
        st.markdown("### 🔍 Clustering Analysis")
        
        target_col = target_cols[0]
        
        # Cluster distribution
        StatisticsDisplay._show_class_distribution(df, target_col, title="Cluster Distribution")
        
        # Feature analysis
        feature_df = df.drop(columns=target_cols)
        StatisticsDisplay._show_statistical_summary(feature_df)
        
        # Visualizations
        StatisticsDisplay._show_feature_visualizations(df, target_col)
    
    @staticmethod
    def _show_regression_analysis(df: pd.DataFrame, target_cols: List[str]) -> None:
        """Show analysis for regression data."""
        st.markdown("### 📈 Regression Analysis")
        
        # Feature analysis
        feature_df = df.drop(columns=target_cols)
        StatisticsDisplay._show_statistical_summary(df)
        
        # Correlation analysis
        PlottingTools.plot_correlation_heatmap(df)
    
    @staticmethod
    def _show_generic_analysis(df: pd.DataFrame) -> None:
        """Show generic analysis for unknown data type."""
        st.markdown("### 📊 General Analysis")
        
        StatisticsDisplay._show_statistical_summary(df)
        PlottingTools.plot_correlation_heatmap(df)
    
    @staticmethod
    def _show_class_distribution(
        df: pd.DataFrame,
        target_col: str,
        title: str = "Class Distribution"
    ) -> None:
        """Display class/cluster distribution."""
        st.markdown(f"#### {title}")
        
        class_counts = df[target_col].value_counts()
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.dataframe(class_counts.to_frame("Count"))
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 8))
            
            def autopct_format(pct, allvals):
                absolute = int(round(pct/100.*sum(allvals)))
                return f"{absolute}\n({pct:.1f}%)"
            
            wedges, texts, autotexts = ax.pie(
                class_counts,
                labels=class_counts.index,
                autopct=lambda pct: autopct_format(pct, class_counts),
                startangle=90,
                colors=sns.color_palette("Set2", len(class_counts))
            )
            
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
            
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.axis("equal")
            
            st.pyplot(fig)
            DownloadManager.download_plot(fig, "pie_chart")
            plt.close(fig)
    
    @staticmethod
    def _show_statistical_summary(df: pd.DataFrame) -> None:
        """Display statistical summary."""
        st.markdown("#### 📏 Statistical Summary")
        
        numeric_df = df.select_dtypes(include=['number'])
        
        if numeric_df.empty:
            st.info("No numerical features found.")
            return
        
        st.dataframe(numeric_df.describe().round(3))
    
    @staticmethod
    def _show_feature_visualizations(df: pd.DataFrame, hue: str) -> None:
        """Show interactive feature visualizations."""
        st.markdown("#### 📊 Feature Visualizations")
        
        features = [col for col in df.columns if col != hue]
        
        if not features:
            st.warning("No features available for visualization.")
            return
        
        # Feature selection
        selected_features = st.multiselect(
            "Select features to visualize",
            options=features,
            default=features[:min(2, len(features))],
            key="viz_features"
        )
        
        if not selected_features:
            return
        
        # Plot type selection
        st.markdown("**Select Plot Types:**")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            show_hist = st.checkbox("📊 Histogram + KDE", value=True)
        with col2:
            show_class_hist = st.checkbox("🎨 Histogram by Class", value=True)
        with col3:
            show_box = st.checkbox("📦 Boxplot", value=False)
        with col4:
            show_scatter = st.checkbox("🔵 Scatter Plot", value=False)
        
        # Show plots
        if show_hist:
            st.markdown("##### 📊 Histogram + KDE")
            PlottingTools.plot_features_in_grid(
                df, selected_features,
                PlottingTools.plot_histogram,
                n_cols=2
            )
        
        if show_class_hist:
            st.markdown("##### 🎨 Histogram by Class")
            PlottingTools.plot_features_in_grid(
                df, selected_features,
                PlottingTools.plot_histogram_by_class,
                n_cols=2, hue=hue
            )
        
        if show_box:
            st.markdown("##### 📦 Boxplot by Class")
            PlottingTools.plot_features_in_grid(
                df, selected_features,
                PlottingTools.plot_boxplot,
                n_cols=2, hue=hue
            )
        
        if show_scatter:
            st.markdown("##### 🔵 Scatter Plot")
            scatter_features = st.multiselect(
                "Select exactly 2 features",
                options=features,
                default=features[:min(2, len(features))],
                key="scatter_features"
            )
            
            if len(scatter_features) == 2:
                PlottingTools.plot_scatter(
                    df,
                    scatter_features[0],
                    scatter_features[1],
                    hue=hue
                )
            elif scatter_features:
                st.warning("Please select exactly 2 features for scatter plot.")