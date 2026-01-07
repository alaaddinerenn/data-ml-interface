import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple
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
        
        # Determine data type (with user interaction if needed)
        data_type, target_cols = StatisticsDisplay._determine_data_type_interactive(df)
        
        # Show appropriate analysis based on type
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
    def _determine_data_type_interactive(df: pd.DataFrame) -> Tuple[str, List[str]]:
        """
        Determine dataset type with user interaction if needed.
        
        Returns:
            Tuple of (data_type, target_columns)
        """
        st.markdown("---")
        st.markdown("### 🎯 Dataset Type Detection")
        
        # Try auto-detection
        auto_type, auto_targets = StatisticsDisplay._auto_detect_type(df)
        
        # Show detection mode selector (always visible)
        if auto_type != "unknown":
            detection_mode = st.radio(
                "Select Detection Mode:",
                ["🤖 Auto-Detection", "✍️ Manual Selection"],
                key="detection_mode",
                horizontal=True,
                index=0  # Default to auto
            )
            
            if detection_mode == "🤖 Auto-Detection":
                st.success(f"✅ **Detected:** {auto_type.replace('_', ' ').title()}")
                if auto_targets:
                    st.info(f"📌 **Target Column(s):** {', '.join(auto_targets)}")
                return auto_type, auto_targets
        else:
            st.warning("⚠️ Could not auto-detect dataset type.")
        
        # Manual selection
        st.markdown("---")
        return StatisticsDisplay._manual_type_selection(df)
    
    @staticmethod
    def _auto_detect_type(df: pd.DataFrame) -> Tuple[str, List[str]]:
        """
        Automatically detect dataset type using heuristics.
        
        Returns:
            Tuple of (data_type, target_columns)
        """
        columns_lower = [col.lower() for col in df.columns]
        
        # Pattern matching (case-insensitive)
        
        # 1. Classification patterns
        classification_patterns = ['label', 'class', 'category', 'species', 'type', 'quality']
        for pattern in classification_patterns:
            matches = [col for col, col_lower in zip(df.columns, columns_lower) if pattern in col_lower]
            if matches:
                # Verify it's categorical
                target_col = matches[0]
                n_unique = df[target_col].nunique()
                if 2 <= n_unique <= 20:  # Reasonable number of classes
                    return "classification", [target_col]
        
        # 2. Clustering patterns
        clustering_patterns = ['cluster', 'group', 'segment']
        for pattern in clustering_patterns:
            matches = [col for col, col_lower in zip(df.columns, columns_lower) if pattern in col_lower]
            if matches:
                return "clustering", [matches[0]]
        
        # 3. Regression patterns
        regression_patterns = ['target', 'price', 'value', 'score', 'rating', 'amount', 'salary', 'income']
        for pattern in regression_patterns:
            matches = [col for col, col_lower in zip(df.columns, columns_lower) 
                      if pattern in col_lower and df[col].dtype in ['int64', 'float64']]
            if matches:
                # Check if continuous
                if df[matches[0]].nunique() > 20:
                    return "regression", [matches[0]]
        
        # 4. Multi-target regression (multiple 'target_' columns)
        target_cols = [col for col in df.columns if col.lower().startswith('target')]
        if len(target_cols) >= 2:
            return "regression", target_cols
        
        # 5. Smart heuristics (no pattern match)
        # Check last column
        last_col = df.columns[-1]
        n_unique = df[last_col].nunique()
        dtype = df[last_col].dtype
        
        # If last column is categorical with few unique values → likely classification
        if dtype == 'object' or (dtype in ['int64', 'float64'] and 2 <= n_unique <= 10):
            return "classification", [last_col]
        
        # If last column is numerical with many unique values → likely regression
        if dtype in ['int64', 'float64'] and n_unique > 20:
            return "regression", [last_col]
        
        return "unknown", []
    
    @staticmethod
    def _manual_type_selection(df: pd.DataFrame) -> Tuple[str, List[str]]:
        """
        Allow user to manually specify dataset type and target.
        
        Returns:
            Tuple of (data_type, target_columns)
        """
        data_type = st.selectbox(
            "📊 Select Dataset Type",
            ["Generic (No specific task)", "Classification", "Regression", "Clustering"],
            key="manual_data_type"
        )
        
        target_cols = []
        
        if data_type == "Classification":
            target_col = st.selectbox(
                "🎯 Select Target Column (Class/Label)",
                options=df.columns,
                key="manual_class_target"
            )
            target_cols = [target_col] if target_col else []
            data_type = "classification"
        
        elif data_type == "Regression":
            allow_multi = st.checkbox(
                "Multiple Targets (Multi-output Regression)",
                value=False,
                key="manual_multi_target"
            )
            
            if allow_multi:
                target_cols = st.multiselect(
                    "🎯 Select Target Columns",
                    options=df.columns,
                    key="manual_reg_targets"
                )
            else:
                target_col = st.selectbox(
                    "🎯 Select Target Column",
                    options=df.columns,
                    key="manual_reg_target"
                )
                target_cols = [target_col] if target_col else []
            
            data_type = "regression"
        
        elif data_type == "Clustering":
            target_col = st.selectbox(
                "🔍 Select Cluster Column (if already clustered)",
                options=["None"] + list(df.columns),
                key="manual_cluster_target"
            )
            target_cols = [target_col] if target_col != "None" else []
            data_type = "clustering"
        
        else:
            data_type = "generic"
        
        return data_type, target_cols
    
    @staticmethod
    def _show_classification_analysis(df: pd.DataFrame, target_cols: List[str]) -> None:
        """Show analysis for classification data."""
        st.markdown("### 🎯 Classification Analysis")
        
        if not target_cols:
            st.warning("No target column specified.")
            StatisticsDisplay._show_generic_analysis(df)
            return
        
        target_col = target_cols[0]
        
        # Class distribution
        StatisticsDisplay._show_class_distribution(df, target_col)
        
        # Feature analysis
        feature_df = df.drop(columns=target_cols)
        StatisticsDisplay._show_statistical_summary(feature_df)
        
        # Correlation matrix (if numerical features exist)
        StatisticsDisplay._show_correlation_if_applicable(df, target_cols)
        
        # Visualizations
        StatisticsDisplay._show_feature_visualizations(df, target_col)
    
    @staticmethod
    def _show_clustering_analysis(df: pd.DataFrame, target_cols: List[str]) -> None:
        """Show analysis for clustering data."""
        st.markdown("### 🔍 Clustering Analysis")
        
        if not target_cols:
            st.info("ℹ️ No cluster column found. Showing generic analysis.")
            StatisticsDisplay._show_generic_analysis(df)
            return
        
        target_col = target_cols[0]
        
        # Cluster distribution
        StatisticsDisplay._show_class_distribution(df, target_col, title="Cluster Distribution")
        
        # Feature analysis
        feature_df = df.drop(columns=target_cols)
        StatisticsDisplay._show_statistical_summary(feature_df)
        
        # Correlation matrix
        StatisticsDisplay._show_correlation_if_applicable(df, target_cols)
        
        # Visualizations
        StatisticsDisplay._show_feature_visualizations(df, target_col)
    
    @staticmethod
    def _show_regression_analysis(df: pd.DataFrame, target_cols: List[str]) -> None:
        """Show analysis for regression data."""
        st.markdown("### 📈 Regression Analysis")
        
        if not target_cols:
            st.warning("No target column specified.")
            StatisticsDisplay._show_generic_analysis(df)
            return
        
        # Feature analysis
        StatisticsDisplay._show_statistical_summary(df)
        
        # Correlation analysis (always applicable for regression)
        PlottingTools.plot_correlation_heatmap(df)
    
    @staticmethod
    def _show_generic_analysis(df: pd.DataFrame) -> None:
        """Show generic analysis for unknown data type."""
        st.markdown("### 📊 General Analysis")
        
        StatisticsDisplay._show_statistical_summary(df)
        
        # Correlation matrix (if numerical features exist)
        StatisticsDisplay._show_correlation_if_applicable(df, exclude_cols=[])
    
    @staticmethod
    def _show_correlation_if_applicable(
        df: pd.DataFrame,
        exclude_cols: List[str] = None
    ) -> None:
        """
        Show correlation heatmap if dataset has numerical features.
        
        Args:
            df: DataFrame to analyze
            exclude_cols: Columns to exclude from correlation (e.g., target/label)
        """
        exclude_cols = exclude_cols or []
        
        # Get numerical columns (excluding target/label/cluster)
        feature_df = df.drop(columns=exclude_cols, errors='ignore')
        numeric_cols = feature_df.select_dtypes(include=['number']).columns.tolist()
        
        # Check if we have enough numerical features
        if len(numeric_cols) < 2:
            if len(numeric_cols) == 1:
                st.info("ℹ️ Only one numerical feature found. Correlation matrix requires at least 2 features.")
            else:
                st.info("ℹ️ No numerical features found for correlation analysis.")
            return
        
        st.markdown("---")
        
        # ✅ Feature selection if more than 15 features
        if len(numeric_cols) > 15:
            st.markdown("### 🔥 Correlation Heatmap")
            st.warning(
                f"⚠️ Dataset has **{len(numeric_cols)}** numerical features. "
                "Please select up to **15 features** for optimal visualization."
            )
            
            selected_features = st.multiselect(
                "Select features for correlation analysis:",
                options=numeric_cols,
                default=numeric_cols[:15],  # Default: first 15
                max_selections=15,
                key="correlation_features",
                help="Maximum 15 features can be selected for clear visualization"
            )
            
            if not selected_features:
                st.info("💡 Please select at least 2 features to display correlation matrix.")
                return
            
            if len(selected_features) < 2:
                st.warning("Please select at least 2 features for correlation analysis.")
                return
            
            # Use selected features
            correlation_df = feature_df[selected_features]
        else:
            # Use all features if <= 15
            correlation_df = feature_df
        
        # Plot correlation heatmap
        PlottingTools.plot_correlation_heatmap(correlation_df)
    
    @staticmethod
    def _show_class_distribution(
        df: pd.DataFrame,
        target_col: str,
        title: str = "Class Distribution"
    ) -> None:
        """Display class/cluster distribution."""
        st.markdown(f"#### {title}")
        
        class_counts = df[target_col].value_counts()
        
        st.dataframe(class_counts.to_frame("Count"))
    
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