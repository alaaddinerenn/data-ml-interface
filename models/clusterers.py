import streamlit as st
import pandas as pd
from typing import Dict, Any, List
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

from .base import BaseClusterer


# KMEANS CLUSTERING
class KMeansModel(BaseClusterer):
    """KMeans Clustering implementation."""
    
    def __init__(self):
        super().__init__("KMeans Clustering", "kmeans_results")
    
    def needs_scaling(self) -> bool:
        """KMeans requires scaling (distance-based)."""
        return True
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get KMeans specific parameters."""
        
        # 1. SCALING OPTIONS FIRST
        st.markdown("#### 📊 Scaling Options")
        st.info("🎯 KMeans is distance-based and requires feature scaling")
        
        scaler_option = st.selectbox(
            "Feature Scaling Method",
            ["StandardScaler (Z-Score)", "MinMaxScaler", "MaxAbsScaler"],
            index=0,  # Default to StandardScaler
            key=f"{self.session_key}_scaler",
            help="Clustering algorithms require all features to be on the same scale"
        )
        
        st.markdown("---")
        
        # 2. KMEANS PARAMETERS
        st.markdown("#### ⚙️ KMeans Parameters")
        
        n_clusters = st.slider(
            "Number of Clusters (k)",
            2, 10, 3,
            key=f"{self.session_key}_n_clusters",
            help="Number of clusters to form"
        )
        
        n_init = st.number_input(
            "Number of Initializations",
            min_value=1,
            max_value=50,
            value=10,
            help="Number of times the k-means algorithm will run with different centroid seeds",
            key=f"{self.session_key}_n_init"
        )
        
        max_iter = st.number_input(
            "Max Iterations",
            min_value=100,
            max_value=1000,
            value=300,
            step=50,
            help="Maximum number of iterations per run",
            key=f"{self.session_key}_max_iter"
        )
        
        return {
            'n_clusters': n_clusters,
            'n_init': int(n_init),
            'max_iter': int(max_iter),
            'random_state': 42
        }
    
    def create_model(self, params: Dict[str, Any]):
        """Create KMeans model."""
        return KMeans(**params)
    
    def get_analysis_options(self) -> List[str]:
        """Get available analysis options."""
        return [
            "Cluster Distribution",
            "Silhouette Analysis",
            "PCA 2D Plot",
            "Feature Boxplots",
            "Elbow Curve"
        ]
    
    def show_model_specific_analysis(
        self,
        results: Dict,
        options: List[str]
    ) -> None:
        """Show KMeans specific visualizations."""
        if "Elbow Curve" in options:
            self.analysis_tools.show_elbow_curve(
                results["X"],
                range(2, 11),
                key_prefix=self.session_key
            )


# DBSCAN CLUSTERING
class DBSCANModel(BaseClusterer):
    """DBSCAN Clustering implementation."""
    
    def __init__(self):
        super().__init__("DBSCAN Clustering", "dbscan_results")
    
    def needs_scaling(self) -> bool:
        """DBSCAN requires scaling (distance-based)."""
        return True
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get DBSCAN specific parameters."""
        
        # 1. SCALING OPTIONS FIRST
        st.markdown("#### 📊 Scaling Options")
        st.info("🎯 DBSCAN is distance-based and requires feature scaling")
        
        scaler_option = st.selectbox(
            "Feature Scaling Method",
            ["StandardScaler (Z-Score)", "MinMaxScaler", "MaxAbsScaler"],
            index=0,
            key=f"{self.session_key}_scaler",
            help="DBSCAN uses distance metrics and requires scaled features"
        )
        
        st.markdown("---")
        
        # 2. DBSCAN PARAMETERS
        st.markdown("#### ⚙️ DBSCAN Parameters")
        
        eps = st.slider(
            "Epsilon (eps) - Maximum distance between samples",
            0.1, 5.0, 0.5, 0.1,
            key=f"{self.session_key}_eps",
            help="Maximum distance between two samples for one to be considered in the neighborhood of the other"
        )
        
        min_samples = st.slider(
            "Minimum Samples - Min points to form a cluster",
            2, 20, 5,
            key=f"{self.session_key}_min_samples",
            help="Number of samples in a neighborhood for a point to be a core point"
        )
        
        metric = st.selectbox(
            "Distance Metric",
            ["euclidean", "manhattan", "chebyshev", "minkowski"],
            key=f"{self.session_key}_metric",
            help="Distance metric to use"
        )
        
        return {
            'eps': eps,
            'min_samples': min_samples,
            'metric': metric
        }
    
    def create_model(self, params: Dict[str, Any]):
        """Create DBSCAN model."""
        return DBSCAN(**params)
    
    def get_analysis_options(self) -> List[str]:
        """Get available analysis options."""
        return [
            "Cluster Distribution",
            "Silhouette Analysis",
            "PCA 2D Plot",
            "Feature Boxplots",
            "Noise Analysis"
        ]
    
    def calculate_metrics(self, X, labels):
        """Override to handle noise points (-1 label)."""
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        st.write(f"**Number of clusters:** {n_clusters}")
        st.write(f"**Number of noise points:** {n_noise} ({n_noise/len(labels)*100:.2f}%)")
        
        # Silhouette score (exclude noise points)
        if n_clusters > 1:
            from sklearn.metrics import silhouette_score
            mask = labels != -1
            if mask.sum() > 0:
                score = silhouette_score(X[mask], labels[mask])
                st.metric("Silhouette Score (excluding noise)", f"{score:.4f}")
        
        return {}
    
    def show_model_specific_analysis(
        self,
        results: Dict,
        options: List[str]
    ) -> None:
        """Show DBSCAN specific visualizations."""
        if "Noise Analysis" in options:
            import matplotlib.pyplot as plt
            import numpy as np
            from utils import DownloadManager
            
            st.markdown("### 🔍 **Noise Points Analysis**")
            
            labels = results["labels"]
            n_noise = list(labels).count(-1)
            n_clustered = len(labels) - n_noise
            
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.bar(["Clustered", "Noise"], [n_clustered, n_noise], 
                  color=["#32CD32", "#FF6347"])
            ax.set_ylabel("Count")
            ax.set_title("Clustered vs Noise Points")
            
            for i, v in enumerate([n_clustered, n_noise]):
                ax.text(i, v + 5, str(v), ha='center', va='bottom', fontweight='bold')
            
            st.pyplot(fig)
            DownloadManager.download_plot(fig, "noise_analysis")
            plt.close(fig)


# AGGLOMERATIVE CLUSTERING
class AgglomerativeModel(BaseClusterer):
    """Agglomerative Hierarchical Clustering implementation."""
    
    def __init__(self):
        super().__init__("Agglomerative Clustering", "agg_results")
    
    def needs_scaling(self) -> bool:
        """Agglomerative requires scaling (distance-based)."""
        return True
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get Agglomerative specific parameters."""
        
        # 1. SCALING OPTIONS FIRST
        st.markdown("#### 📊 Scaling Options")
        st.info("🎯 Agglomerative is distance-based and requires feature scaling")
        
        scaler_option = st.selectbox(
            "Feature Scaling Method",
            ["StandardScaler (Z-Score)", "MinMaxScaler", "MaxAbsScaler"],
            index=0,
            key=f"{self.session_key}_scaler",
            help="Hierarchical clustering uses distance metrics and requires scaled features"
        )
        
        st.markdown("---")
        
        # 2. AGGLOMERATIVE PARAMETERS
        st.markdown("#### ⚙️ Agglomerative Parameters")
        
        n_clusters = st.slider(
            "Number of Clusters",
            2, 10, 3,
            key=f"{self.session_key}_n_clusters",
            help="Number of clusters to find"
        )
        
        linkage = st.selectbox(
            "Linkage Method",
            ["ward", "complete", "average", "single"],
            key=f"{self.session_key}_linkage",
            help="Which linkage criterion to use"
        )
        
        metric = st.selectbox(
            "Distance Metric",
            ["euclidean", "manhattan", "cosine"],
            key=f"{self.session_key}_metric",
            help="Distance metric to use"
        )
        
        # Ward linkage only works with euclidean
        if linkage == "ward" and metric != "euclidean":
            st.warning("⚠ Ward linkage requires Euclidean metric. Automatically set to Euclidean.")
            metric = "euclidean"
        
        return {
            'n_clusters': n_clusters,
            'linkage': linkage,
            'metric': metric
        }
    
    def create_model(self, params: Dict[str, Any]):
        """Create Agglomerative model."""
        return AgglomerativeClustering(**params)
    
    def get_analysis_options(self) -> List[str]:
        """Get available analysis options."""
        return [
            "Cluster Distribution",
            "Silhouette Analysis",
            "PCA 2D Plot",
            "Feature Boxplots",
            "Dendrogram"
        ]
    
    def show_model_specific_analysis(
        self,
        results: Dict,
        options: List[str]
    ) -> None:
        """Show Agglomerative specific visualizations."""
        if "Dendrogram" in options:
            import matplotlib.pyplot as plt
            from scipy.cluster.hierarchy import dendrogram, linkage
            from utils import DownloadManager

            st.markdown("### 🌳 **Dendrogram**")

            # Perform hierarchical clustering
            Z = linkage(results["X"], method=results["model_params"]["linkage"])
            
            fig, ax = plt.subplots(figsize=(12, 6))
            dendrogram(Z, ax=ax, truncate_mode='lastp', p=30)
            ax.set_xlabel("Sample Index or (Cluster Size)")
            ax.set_ylabel("Distance")
            ax.set_title(f"Dendrogram ({results['model_params']['linkage']} linkage)", 
                        fontsize=14, fontweight='bold')
            
            st.pyplot(fig)
            DownloadManager.download_plot(fig, "dendrogram")
            plt.close(fig)


# PUBLIC API
def kmeans_page(df: pd.DataFrame) -> None:
    """Entry point for KMeans page."""
    model = KMeansModel()
    model.page(df)


def dbscan_page(df: pd.DataFrame) -> None:
    """Entry point for DBSCAN page."""
    model = DBSCANModel()
    model.page(df)


def agglomerative_page(df: pd.DataFrame) -> None:
    """Entry point for Agglomerative Clustering page."""
    model = AgglomerativeModel()
    model.page(df)