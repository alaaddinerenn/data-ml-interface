import streamlit as st
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from models.utils import encode_features
from models.classification_utils import AnalysisTools


class BaseClassifier(ABC):
    """
    Abstract base class for all classifiers.
    
    This class provides a common interface and shared functionality for all ML classifiers.
    Child classes must implement abstract methods for model-specific behavior.
    
    Attributes:
        model_name (str): Display name of the model
        session_key (str): Unique key for storing results in session state
        analysis_tools (AnalysisTools): Instance of analysis utilities
    """
    
    def __init__(self, model_name: str, session_key: str):
        """
        Initialize the base classifier.
        
        Args:
            model_name: Display name for the model (e.g., "Decision Tree Classifier")
            session_key: Unique key for session state storage (e.g., "dt_results")
        """
        self.model_name = model_name
        self.session_key = session_key
        self.analysis_tools = AnalysisTools()
    
    def get_common_params(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Display UI elements and collect common training parameters.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary containing common parameters like target, features, encoding type, etc.
        """
        st.subheader("🔹 Model Settings")
        
        # Target selection
        target = st.selectbox(
            "Select Target Column",
            options=df.columns,
            index=list(df.columns).index('label') if 'label' in df.columns else 0,
            key=f"{self.session_key}_target"
        )
        
        # Feature selection
        features = st.multiselect(
            "Select Features for the Model",
            options=[col for col in df.columns if col != target],
            default=[col for col in df.columns if col != target],
            key=f"{self.session_key}_features"
        )
        
        # Encoding
        encoding_type = st.radio(
            "Encoding Type",
            ["One-Hot Encoding", "Label Encoding"],
            key=f"{self.session_key}_encoding"
        )
        
        # Train-test split parameters
        test_size = st.slider(
            "Test Set Ratio",
            0.1, 0.5, 0.2, 0.05,
            key=f"{self.session_key}_test_size"
        )
        
        random_seed = st.number_input(
            "Random Seed",
            value=42,
            step=1,
            key=f"{self.session_key}_seed"
        )
        
        shuffle_data = st.checkbox(
            "Shuffle Data",
            value=True,
            key=f"{self.session_key}_shuffle"
        )
        
        use_stratify = st.checkbox(
            "Stratify (Preserve Class Distribution)",
            value=True,
            key=f"{self.session_key}_stratify"
        )
        
        return {
            'target': target,
            'features': features,
            'encoding_type': encoding_type,
            'test_size': test_size,
            'random_seed': int(random_seed),
            'shuffle_data': shuffle_data,
            'use_stratify': use_stratify
        }
    
    @abstractmethod
    def get_model_params(self) -> Dict[str, Any]:
        """
        Display UI elements and collect model-specific parameters.
        
        Must be implemented by child classes.
        
        Returns:
            Dictionary containing model-specific parameters
        """
        pass
    
    @abstractmethod
    def create_model(self, params: Dict[str, Any]):
        """
        Create and return the model instance.
        
        Must be implemented by child classes.
        
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
        
        Must be implemented by child classes.
        
        Returns:
            List of analysis option names
        """
        pass
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        common_params: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str]]:
        """
        Prepare and split data for training.
        
        Args:
            df: Input DataFrame
            common_params: Common parameters from get_common_params()
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test, features)
            
        Raises:
            ValueError: If no features are selected
        """
        if not common_params['features']:
            raise ValueError("You must select at least one feature.")
        
        # Encoding
        df_encoded = encode_features(
            df,
            common_params['encoding_type'],
            target_col=common_params['target']
        )
        
        # Update features list after encoding (for one-hot encoding)
        if common_params['encoding_type'] == "One-Hot Encoding":
            features = [col for col in df_encoded.columns 
                       if col != common_params['target']]
        else:
            features = common_params['features']
        
        X = df_encoded[features]
        y = df[common_params['target']]
        
        # Train-test split
        stratify_param = y if common_params['use_stratify'] else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=common_params['test_size'],
            random_state=common_params['random_seed'],
            shuffle=common_params['shuffle_data'],
            stratify=stratify_param
        )
        
        return X_train, X_test, y_train, y_test, features
    
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
                X_train, X_test, y_train, y_test, features = self.prepare_data(
                    df, common_params
                )
                
                # Create and train model
                model = self.create_model(model_params)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                # Calculate accuracy
                accuracy = accuracy_score(y_test, y_pred)
                
                # Save results to session state
                st.session_state[self.session_key] = {
                    "model": model,
                    "X_test": X_test,
                    "X_train": X_train,
                    "y_test": y_test,
                    "y_train": y_train,
                    "y_pred": y_pred,
                    "features": features,
                    "model_params": model_params
                }
                
                # Show success message
                st.success(f"✅ {self.model_name} trained successfully!")
                st.write(f"**Accuracy:** {accuracy:.4f}")
                
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
            "Select analyses you want to visualize",
            self.get_analysis_options(),
            default=["Classification Report"],
            key=f"{self.session_key}_analysis"
        )
        
        # Common analyses
        if "Classification Report" in analysis_options:
            self.analysis_tools.show_classification_report(
                results["y_test"],
                results["y_pred"]
            )
        
        if "Confusion Matrix" in analysis_options:
            self.analysis_tools.show_confusion_matrix(
                results["y_test"],
                results["y_pred"],
                key_prefix=self.session_key
            )
        
        if "ROC Curve" in analysis_options:
            self.analysis_tools.show_roc_curve(
                results["y_test"],
                results["model"],
                results["X_test"],
                key_prefix=self.session_key
            )
        
        if "Decision Regions" in analysis_options:
            self._show_decision_regions(results)
        
        if "Learning Curve" in analysis_options:
            self.analysis_tools.show_learning_curve(
                results["model"],
                results["X_train"],
                results["y_train"],
                f"Learning Curve ({self.model_name})"
            )
        
        # Model-specific analyses
        self.show_model_specific_analysis(results, analysis_options)
    
    def _show_decision_regions(self, results: Dict) -> None:
        """
        Display decision regions plot.
        
        Args:
            results: Results dictionary from session state
        """
        model_class = type(results["model"])
        self.analysis_tools.show_decision_regions(
            results["X_train"],
            results["y_train"],
            results["X_test"],
            results["y_test"],
            model_class,
            results["model_params"],
            key_prefix=self.session_key
        )
    
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