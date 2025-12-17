import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.cluster import KMeans
from matplotlib.colors import ListedColormap
from typing import List

from utils import DownloadManager


class ClusteringAnalysisTools:
    """
    Utility class containing common analysis and visualization methods for clustering.
    """
    
    # Consistent color palette for clusters
    CLUSTER_COLORS = ["#1E90FF", "#32CD32", "#FFA500", "#FF6347", "#9370DB", 
                      "#20B2AA", "#FFD700", "#FF69B4", "#8B4513", "#4682B4"]
    
    @staticmethod
    def show_cluster_distribution(
        labels: np.ndarray,
        key_prefix: str = "cluster"
    ) -> None:
        """
        Display cluster distribution as table and pie chart.
        
        Args:
            labels: Cluster labels
            key_prefix: Prefix for unique widget keys
        """
        st.markdown("### 📊 **Cluster Distribution**")
        
        unique, counts = np.unique(labels, return_counts=True)
        dist_df = pd.DataFrame({"Cluster": unique, "Count": counts})
        dist_df["Percentage"] = (dist_df["Count"] / dist_df["Count"].sum() * 100).round(2)
        
        st.dataframe(dist_df)
        
        # Pie chart
        fig, ax = plt.subplots(figsize=(8, 6))
        
        def autopct_format(pct, allvals):
            absolute = int(round(pct/100.*sum(allvals)))
            return f"{absolute}\n({pct:.1f}%)"
        
        colors = ClusteringAnalysisTools.CLUSTER_COLORS[:len(unique)]
        wedges, texts, autotexts = ax.pie(
            counts,
            labels=[f"Cluster {i}" for i in unique],
            autopct=lambda pct: autopct_format(pct, counts),
            startangle=90,
            colors=colors
        )
        
        # Make percentage text more readable
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(10)
            autotext.set_weight('bold')
        
        ax.set_title("Cluster Distribution", fontsize=14, fontweight='bold')
        ax.axis("equal")
        
        st.pyplot(fig)
        DownloadManager.download_plot(fig, "cluster_distribution_pie")
        plt.close(fig)
    
    @staticmethod
    def show_silhouette_analysis(
        X: np.ndarray,
        labels: np.ndarray,
        key_prefix: str = "cluster"
    ) -> None:
        """
        Display silhouette score and silhouette plot.
        
        Args:
            X: Feature matrix
            labels: Cluster labels
            key_prefix: Prefix for unique widget keys
        """
        st.markdown("### 📈 **Silhouette Analysis**")
        
        n_clusters = len(np.unique(labels))
        
        if n_clusters < 2:
            st.warning("Silhouette analysis requires at least 2 clusters.")
            return
        
        # Overall silhouette score
        score = silhouette_score(X, labels)
        st.metric("Average Silhouette Score", f"{score:.4f}")
        
        # Silhouette plot
        sample_scores = silhouette_samples(X, labels)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        y_lower = 10
        colors = ClusteringAnalysisTools.CLUSTER_COLORS[:n_clusters]
        
        for i in range(n_clusters):
            # Get silhouette values for cluster i
            ith_cluster_silhouette_values = sample_scores[labels == i]
            ith_cluster_silhouette_values.sort()
            
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
            
            color = colors[i]
            ax.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7
            )
            
            # Label the silhouette plots with their cluster numbers at the middle
            ax.text(-0.05, y_lower + 0.5 * size_cluster_i, f"C{i}")
            
            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for spacing
        
        # Average silhouette line
        ax.axvline(score, color="red", linestyle="--", linewidth=2, 
                  label=f"Average Score: {score:.3f}")
        
        ax.set_xlabel("Silhouette Coefficient Values", fontsize=12)
        ax.set_ylabel("Cluster Label", fontsize=12)
        ax.set_title("Silhouette Plot for Each Cluster", fontsize=14, fontweight='bold')
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        DownloadManager.download_plot(fig, "silhouette_plot")
        plt.close(fig)
        
        # K vs Silhouette Score (if using KMeans-like algorithm)
        st.markdown("### **🔍 K vs Silhouette Score**")

        k_range = range(2, min(11, len(X)))
        sil_scores = []
        
        with st.spinner("Calculating silhouette scores for different k values..."):
            for k in k_range:
                km = KMeans(n_clusters=k, random_state=42, n_init=10)
                temp_labels = km.fit_predict(X)
                sil_scores.append(silhouette_score(X, temp_labels))
        
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        ax2.plot(k_range, sil_scores, marker="o", linewidth=2, markersize=8)
        ax2.set_xlabel("Number of Clusters (k)", fontsize=12)
        ax2.set_ylabel("Average Silhouette Score", fontsize=12)
        ax2.set_title("K vs Silhouette Score", fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Highlight best k
        best_k = list(k_range)[np.argmax(sil_scores)]
        ax2.axvline(best_k, color='red', linestyle='--', alpha=0.5, 
                   label=f'Best k={best_k}')
        ax2.legend()
        
        st.pyplot(fig2)
        DownloadManager.download_plot(fig2, "k_vs_silhouette_score")
        plt.close(fig2)
    
    @staticmethod
    def show_pca_plot(
        X: np.ndarray,
        labels: np.ndarray,
        feature_names: List[str],
        key_prefix: str = "cluster"
    ) -> None:
        """
        Display PCA 2D visualization of clusters.
        
        Args:
            X: Feature matrix
            labels: Cluster labels
            feature_names: Names of features
            key_prefix: Prefix for unique widget keys
        """
        st.markdown("### 🗺 **PCA 2D Cluster Visualization**")

        # Perform PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 7))
        
        n_clusters = len(np.unique(labels))
        colors = ClusteringAnalysisTools.CLUSTER_COLORS[:n_clusters]
        cmap = ListedColormap(colors)
        
        scatter = ax.scatter(
            X_pca[:, 0],
            X_pca[:, 1],
            c=labels,
            cmap=cmap,
            s=50,
            alpha=0.6,
            edgecolors='k',
            linewidth=0.5
        )
        
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=12)
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=12)
        ax.set_title("Clusters in PCA 2D Space", fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Cluster', fontsize=12)
        
        st.pyplot(fig)
        DownloadManager.download_plot(fig, "pca_clusters")
        plt.close(fig)
        
        # Show explained variance
        st.info(f"📊 Total variance explained by 2 PCs: "
               f"{sum(pca.explained_variance_ratio_)*100:.2f}%")
    
    @staticmethod
    def show_feature_boxplots(
        X: np.ndarray,
        labels: np.ndarray,
        feature_names: List[str],
        key_prefix: str = "cluster"
    ) -> None:
        """
        Display boxplots of features by cluster.
        
        Args:
            X: Feature matrix
            labels: Cluster labels
            feature_names: Names of features
            key_prefix: Prefix for unique widget keys
        """
        st.markdown("### 📦 **Feature Distribution by Cluster**")

        # Create DataFrame
        df_box = pd.DataFrame(X, columns=feature_names)
        df_box["Cluster"] = labels
        
        # Select features to plot
        selected_features = st.multiselect(
            "Select features to visualize",
            feature_names,
            default=feature_names[:min(3, len(feature_names))],
            key=f"{key_prefix}_boxplot_features"
        )
        
        if not selected_features:
            st.warning("Please select at least one feature.")
            return
        
        colors = ClusteringAnalysisTools.CLUSTER_COLORS[:len(np.unique(labels))]
        
        for feature in selected_features:
            fig, ax = plt.subplots(figsize=(10, 5))
            
            # Boxplot
            sns.boxplot(
                data=df_box,
                x="Cluster",
                y=feature,
                palette=colors,
                ax=ax
            )
            
            ax.set_title(f"Distribution of {feature} by Cluster", 
                        fontsize=14, fontweight='bold')
            ax.set_xlabel("Cluster", fontsize=12)
            ax.set_ylabel(feature, fontsize=12)
            ax.grid(True, alpha=0.3, axis='y')
            
            st.pyplot(fig)
            DownloadManager.download_plot(fig, f"boxplot_{feature}")
            plt.close(fig)
    
    @staticmethod
    def show_elbow_curve(
        X: np.ndarray,
        k_range: range = range(2, 11),
        key_prefix: str = "cluster"
    ) -> None:
        """
        Display elbow curve for determining optimal k.
        
        Args:
            X: Feature matrix
            k_range: Range of k values to test
            key_prefix: Prefix for unique widget keys
        """
        st.markdown("### 📉 **Elbow Method**")

        inertias = []
        
        with st.spinner("Calculating inertia for different k values..."):
            for k in k_range:
                km = KMeans(n_clusters=k, random_state=42, n_init=10)
                km.fit(X)
                inertias.append(km.inertia_)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(k_range, inertias, "bo-", linewidth=2, markersize=8)
        ax.set_xlabel("Number of Clusters (k)", fontsize=12)
        ax.set_ylabel("Inertia (Within-Cluster Sum of Squares)", fontsize=12)
        ax.set_title("Elbow Curve", fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        DownloadManager.download_plot(fig, "elbow_curve")
        plt.close(fig)
        
        # Show inertia values in table
        with st.expander("📋 View Inertia Values"):
            inertia_df = pd.DataFrame({
                "k": list(k_range),
                "Inertia": inertias
            })
            st.dataframe(inertia_df)