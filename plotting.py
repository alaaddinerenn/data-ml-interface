import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pandas as pd
from typing import Optional, List, Callable
from utils import DownloadManager


class PlottingTools:
    """
    Comprehensive plotting utilities for data visualization.
    Provides various plot types for exploratory data analysis.
    """
    
    # Consistent color palette
    DEFAULT_PALETTE = "Set2"
    
    @staticmethod
    def _chunks(lst: List, n: int):
        """Split list into chunks of size n."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]
    
    @staticmethod
    def plot_features_in_grid(
        df: pd.DataFrame,
        selected_features: List[str],
        plot_func: Callable,
        n_cols: int = 2,
        hue: Optional[str] = "label",
        palette: Optional[str] = None
    ) -> None:
        """
        Plot multiple features in a grid layout.
        
        Args:
            df: DataFrame containing the data
            selected_features: List of feature names to plot
            plot_func: Plotting function to use
            n_cols: Number of columns in grid
            hue: Column name for color coding
            palette: Color palette name
        """
        if not selected_features:
            st.warning("No features selected for plotting.")
            return
        
        palette = palette or PlottingTools.DEFAULT_PALETTE
        
        for chunk_features in PlottingTools._chunks(selected_features, n_cols):
            cols = st.columns(len(chunk_features))
            for i, feature in enumerate(chunk_features):
                with cols[i]:
                    plot_func(feature, df, hue, palette)
    
    @staticmethod
    def plot_histogram(
        feature: str,
        df: pd.DataFrame,
        hue: Optional[str] = None,
        palette: Optional[str] = None,
        bins: int = 20,
        kde: bool = True
    ) -> None:
        """
        Plot histogram with optional KDE.
        
        Args:
            feature: Feature name to plot
            df: DataFrame containing the data
            hue: Column for color coding (optional)
            palette: Color palette
            bins: Number of bins
            kde: Show KDE overlay
        """
        st.markdown(f"**{feature}**")
        
        fig, ax = plt.subplots(figsize=(8, 5))
        
        sns.histplot(
            data=df,
            x=feature,
            kde=kde,
            bins=bins,
            ax=ax,
            color='steelblue'
        )
        
        ax.set_title(f"Distribution of {feature}")
        ax.set_xlabel(feature)
        ax.set_ylabel("Frequency")
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        DownloadManager.download_plot(fig, "histogram", feature.strip())
        plt.close(fig)
    
    @staticmethod
    def plot_histogram_by_class(
        feature: str,
        df: pd.DataFrame,
        hue: Optional[str] = "label",
        palette: Optional[str] = None,
        bins: int = 20,
        kde: bool = True
    ) -> None:
        """
        Plot histogram colored by class/category.
        
        Args:
            feature: Feature name to plot
            df: DataFrame containing the data
            hue: Column for color coding
            palette: Color palette
            bins: Number of bins
            kde: Show KDE overlay
        """
        st.markdown(f"**{feature}**")
        
        if hue not in df.columns:
            st.warning(f"Column '{hue}' not found. Plotting without class separation.")
            PlottingTools.plot_histogram(feature, df, bins=bins, kde=kde)
            return
        
        fig, ax = plt.subplots(figsize=(8, 5))
        
        sns.histplot(
            data=df,
            x=feature,
            hue=hue,
            kde=kde,
            bins=bins,
            ax=ax,
            palette=palette or PlottingTools.DEFAULT_PALETTE,
            alpha=0.6
        )
        
        ax.set_title(f"Distribution of {feature} by {hue}")
        ax.set_xlabel(feature)
        ax.set_ylabel("Frequency")
        ax.grid(True, alpha=0.3)
        ax.legend(title=hue)
        
        st.pyplot(fig)
        DownloadManager.download_plot(fig, "histogram_by_class", feature.strip())
        plt.close(fig)
    
    @staticmethod
    def plot_boxplot(
        feature: str,
        df: pd.DataFrame,
        hue: Optional[str] = "label",
        palette: Optional[str] = None
    ) -> None:
        """
        Plot boxplot for feature distribution.
        
        Args:
            feature: Feature name to plot
            df: DataFrame containing the data
            hue: Column for grouping
            palette: Color palette
        """
        st.markdown(f"**{feature}**")
        
        fig, ax = plt.subplots(figsize=(8, 5))
        
        if hue in df.columns:
            sns.boxplot(
                data=df,
                x=hue,
                y=feature,
                ax=ax,
                palette=palette or PlottingTools.DEFAULT_PALETTE
            )
            ax.set_title(f"Distribution of {feature} by {hue}")
        else:
            sns.boxplot(
                data=df,
                y=feature,
                ax=ax,
                color='steelblue'
            )
            ax.set_title(f"Distribution of {feature}")
        
        ax.set_ylabel(feature)
        ax.grid(True, alpha=0.3, axis='y')
        
        st.pyplot(fig)
        DownloadManager.download_plot(fig, "boxplot", feature.strip())
        plt.close(fig)
    
    @staticmethod
    def plot_scatter(
        df: pd.DataFrame,
        x_feature: str,
        y_feature: str,
        hue: Optional[str] = "label",
        palette: Optional[str] = None,
        size: int = 70,
        alpha: float = 0.8
    ) -> None:
        """
        Plot scatter plot for two features.
        
        Args:
            df: DataFrame containing the data
            x_feature: Feature for x-axis
            y_feature: Feature for y-axis
            hue: Column for color coding
            palette: Color palette
            size: Marker size
            alpha: Transparency
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if hue in df.columns:
            sns.scatterplot(
                data=df,
                x=x_feature,
                y=y_feature,
                hue=hue,
                style=hue,
                palette=palette or PlottingTools.DEFAULT_PALETTE,
                s=size,
                alpha=alpha,
                ax=ax
            )
            ax.legend(title=hue, bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            sns.scatterplot(
                data=df,
                x=x_feature,
                y=y_feature,
                s=size,
                alpha=alpha,
                ax=ax,
                color='steelblue'
            )
        
        ax.set_title(f"Scatter Plot: {x_feature} vs {y_feature}")
        ax.set_xlabel(x_feature)
        ax.set_ylabel(y_feature)
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        DownloadManager.download_plot(fig, "scatter", [x_feature, y_feature])
        plt.close(fig)
    
    @staticmethod
    def plot_correlation_heatmap(
        df: pd.DataFrame,
        figsize: tuple = (12, 10),
        cmap: str = "coolwarm"
    ) -> None:
        """
        Plot correlation heatmap for numerical features.
        
        Args:
            df: DataFrame containing the data
            figsize: Figure size
            cmap: Color map
        """
        # Select only numeric columns
        numeric_df = df.select_dtypes(include=['number'])
        
        if numeric_df.empty:
            st.warning("No numerical features found for correlation analysis.")
            return
        
        st.markdown("### 🔥 Correlation Heatmap")
        
        corr_matrix = numeric_df.corr()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap=cmap,
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            ax=ax
        )
        
        ax.set_title("Feature Correlation Matrix", fontsize=16, fontweight='bold')
        
        st.pyplot(fig)
        DownloadManager.download_plot(fig, "correlation_heatmap")
        plt.close(fig)