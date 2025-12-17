import streamlit as st
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
from sklearn.metrics import silhouette_score

from models.utils import encode_features
from models.clustering_utils import ClusteringAnalysisTools


class BaseClusterer(ABC):
    """
    Abstract base class for all clustering models.
    
    Clustering is unsupervised learning - no target variable needed.
    
    Attributes:
        model_name (str): Display name of the model
        session_key (str): Unique key for storing results in session state
        analysis_tools (ClusteringAnalysisTools): Instance of clustering analysis utilities
    """
    
    def __init__(self, model_name: str, session_key: str):
        """
        Initialize the base clusterer.
        
        Args:
            model_name: Display name for the model (e.g., "KMeans Clustering")
            session_key: Unique key for session state storage (e.g., "kmeans_results")
        """
        self.model_name = model_name
        self.session_key = session_key
        self.analysis_tools = ClusteringAnalysisTools()
    
    def get_common_params(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Display UI elements and collect common clustering parameters.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary containing common parameters
        """
        st.subheader("🔹 Model Settings")
        
        # Feature selection (no target for clustering!)
        features = st.multiselect(
            "Select Features for Clustering",
            options=df.columns,
            default=[col for col in df.columns if 'cluster' not in col.lower()],
            key=f"{self.session_key}_features"
        )
        
        # Encoding
        encoding_type = st.radio(
            "Encoding Type",
            ["One-Hot Encoding", "Label Encoding"],
            key=f"{self.session_key}_encoding"
        )
        
        # Scaling (almost mandatory for clustering)
        scale_data = st.checkbox(
            "Scale Data (StandardScaler)",
            value=True,
            help="Highly recommended for distance-based clustering",
            key=f"{self.session_key}_scale"
        )
        
        # Random seed
        random_seed = st.number_input(
            "Random Seed",
            value=42,
            step=1,
            key=f"{self.session_key}_seed"
        )
        
        return {
            'features': features,
            'encoding_type': encoding_type,
            'scale_data': scale_data,
            'random_seed': int(random_seed)
        }
    
    @abstractmethod
    def get_model_params(self) -> Dict[str, Any]:
        """
        Get model-specific parameters from UI (implemented by child classes).
        
        Returns:
            Dictionary containing model-specific parameters
        """
        pass
    
    @abstractmethod
    def create_model(self, params: Dict[str, Any]):
        """
        Create and return the model instance.
        
        Args:
            params: Model parameters from get_model_params()
            
        Returns:
            Initialized model instance
        """
        pass
    
    @abstractmethod
    def get_analysis_options(self) -> List[str]:
        """
        Get list of available analysis options for this model.
        
        Returns:
            List of analysis option names
        """
        pass
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        common_params: Dict[str, Any]
    ) -> Tuple[np.ndarray, List[str], Optional[Any]]:
        """
        Prepare data for clustering.
        
        Note: Clustering is unsupervised - no y variable!
        
        Args:
            df: Input DataFrame
            common_params: Common parameters from get_common_params()
            
        Returns:
            Tuple of (X, features, scaler)
            
        Raises:
            ValueError: If no features are selected
        """
        if not common_params['features']:
            raise ValueError("You must select at least one feature.")
        
        # Encoding
        df_encoded = encode_features(
            df,
            common_params['encoding_type'],
            target_col=None  # No target in clustering!
        )
        
        # Update features after encoding
        if common_params['encoding_type'] == "One-Hot Encoding":
            features = [col for col in df_encoded.columns 
                       if col in df_encoded.columns]
        else:
            features = common_params['features']
        
        X = df_encoded[features]
        
        # Scaling (highly recommended for clustering)
        scaler = None
        if common_params['scale_data']:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
        else:
            X = X.values
        
        return X, features, scaler
    
    def calculate_metrics(self, X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """
        Calculate and display clustering metrics.
        
        Args:
            X: Feature matrix
            labels: Cluster labels
            
        Returns:
            Dictionary of metrics
        """
        n_clusters = len(np.unique(labels))
        
        metrics = {}
        
        # Silhouette Score (only if k > 1)
        if n_clusters > 1:
            sil_score = silhouette_score(X, labels)
            metrics['silhouette_score'] = sil_score
        else:
            metrics['silhouette_score'] = None
        
        # Inertia (if model has it)
        if hasattr(st.session_state[self.session_key]['model'], 'inertia_'):
            metrics['inertia'] = st.session_state[self.session_key]['model'].inertia_
        
        # Display metrics
        cols = st.columns(len(metrics))
        for i, (key, value) in enumerate(metrics.items()):
            if value is not None:
                cols[i].metric(
                    key.replace('_', ' ').title(),
                    f"{value:.4f}"
                )
        
        return metrics
    
    def train(self, df: pd.DataFrame) -> None:
        """
        Main training workflow.
        
        Args:
            df: Input DataFrame
        """
        common_params = self.get_common_params(df)
        model_params = self.get_model_params()
        
        if st.button("Train Model", key=f"{self.session_key}_train_btn"):
            try:
                # Prepare data
                X, features, scaler = self.prepare_data(df, common_params)
                
                # Create and train model
                model = self.create_model(model_params)
                labels = model.fit_predict(X)
                
                # Save results to session state
                st.session_state[self.session_key] = {
                    "model": model,
                    "X": X,
                    "labels": labels,
                    "features": features,
                    "scaler": scaler,
                    "model_params": model_params,
                    "scaled": common_params['scale_data']
                }
                
                # Show success and metrics
                st.success(f"✅ {self.model_name} trained successfully!")
                
                # Show cluster distribution
                unique, counts = np.unique(labels, return_counts=True)
                st.write("**Cluster Distribution:**")
                dist_df = pd.DataFrame({"Cluster": unique, "Count": counts})
                st.dataframe(dist_df)
                
                # Calculate metrics
                self.calculate_metrics(X, labels)
                
            except Exception as e:
                st.error(f"❌ Error during training: {str(e)}")
    
    def analyze(self) -> None:
        """
        Main analysis workflow.
        """
        if self.session_key not in st.session_state:
            st.warning("⚠ You must train the model first.")
            return
        
        results = st.session_state[self.session_key]
        
        st.subheader("📊 Analysis Tools")
        analysis_options = st.multiselect(
            "Select analyses to visualize",
            self.get_analysis_options(),
            default=["Cluster Distribution"],
            key=f"{self.session_key}_analysis"
        )
        
        # Common analyses
        if "Cluster Distribution" in analysis_options:
            self.analysis_tools.show_cluster_distribution(
                results["labels"],
                key_prefix=self.session_key
            )
        
        if "Silhouette Analysis" in analysis_options:
            self.analysis_tools.show_silhouette_analysis(
                results["X"],
                results["labels"],
                key_prefix=self.session_key
            )
        
        if "PCA 2D Plot" in analysis_options:
            self.analysis_tools.show_pca_plot(
                results["X"],
                results["labels"],
                results["features"],
                key_prefix=self.session_key
            )
        
        if "Feature Boxplots" in analysis_options:
            self.analysis_tools.show_feature_boxplots(
                results["X"],
                results["labels"],
                results["features"],
                key_prefix=self.session_key
            )
        
        # Model-specific analyses
        self.show_model_specific_analysis(results, analysis_options)
    
    def show_model_specific_analysis(
        self,
        results: Dict,
        options: List[str]
    ) -> None:
        """
        Display model-specific analyses.
        
        Override this in child classes for custom analyses.
        
        Args:
            results: Results dictionary from session state
            options: Selected analysis options
        """
        pass
    
    def page(self, df: pd.DataFrame) -> None:
        """
        Main page function that combines training and analysis.
        
        Args:
            df: Input DataFrame
        """
        st.title(self.model_name)
        self.train(df)
        st.markdown("---")
        self.analyze()