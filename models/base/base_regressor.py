import streamlit as st
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from models.utils import encode_features
from models.regression_utils import RegressionAnalysisTools


class BaseRegressor(ABC):
    """
    Abstract base class for all regression models.
    
    Attributes:
        model_name (str): Display name of the model
        session_key (str): Unique key for storing results in session state
        analysis_tools (RegressionAnalysisTools): Instance of regression analysis utilities
    """
    
    def __init__(self, model_name: str, session_key: str):
        """
        Initialize the base regressor.
        
        Args:
            model_name: Display name for the model (e.g., "Linear Regression")
            session_key: Unique key for session state storage (e.g., "linreg_results")
        """
        self.model_name = model_name
        self.session_key = session_key
        self.analysis_tools = RegressionAnalysisTools()
    
    def get_common_params(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Display UI elements and collect common training parameters.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary containing common parameters
        """
        st.subheader("🔹 Model Settings")
        
        # Target selection (single or multi)
        allow_multi_target = self.supports_multi_output()
        
        if allow_multi_target:
            target = st.multiselect(
                "Select Target Variable(s)",
                options=df.columns,
                default=[col for col in df.columns if 'target' in col.lower()],
                key=f"{self.session_key}_target"
            )
        else:
            target = st.selectbox(
                "Select Target Variable",
                options=df.columns,
                index=list(df.columns).index('target') if 'target' in df.columns else 0,
                key=f"{self.session_key}_target"
            )
            target = [target] if target else []
        
        # Feature selection
        features = st.multiselect(
            "Select Features for the Model",
            options=[col for col in df.columns if col not in target],
            default=[col for col in df.columns if col not in target],
            key=f"{self.session_key}_features"
        )
        
        # Encoding
        encoding_type = st.radio(
            "Encoding Type",
            ["One-Hot Encoding", "Label Encoding"],
            key=f"{self.session_key}_encoding"
        )
        
        # Train-test split
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
        
        return {
            'target': target,
            'features': features,
            'encoding_type': encoding_type,
            'test_size': test_size,
            'random_seed': int(random_seed),
            'shuffle_data': shuffle_data
        }
    
    @abstractmethod
    def get_model_params(self) -> Dict[str, Any]:
        """Get model-specific parameters from UI (implemented by child classes)"""
        pass
    
    @abstractmethod
    def create_model(self, params: Dict[str, Any]):
        """Create and return the model instance (implemented by child classes)"""
        pass
    
    @abstractmethod
    def get_analysis_options(self) -> List[str]:
        """Get available analysis options for this model"""
        pass
    
    def supports_multi_output(self) -> bool:
        """
        Override this if model supports multiple output targets.
        
        Returns:
            True if model supports multi-output, False otherwise
        """
        return False
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        common_params: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str], bool]:
        """
        Prepare and split data for training.
        
        Args:
            df: Input DataFrame
            common_params: Common parameters from get_common_params()
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test, features, is_multioutput)
        """
        if not common_params['features']:
            raise ValueError("You must select at least one feature.")
        if not common_params['target']:
            raise ValueError("You must select at least one target variable.")
        
        # Encoding
        df_encoded = encode_features(
            df,
            common_params['encoding_type'],
            target_col=common_params['target']
        )
        
        # Update features after encoding
        if common_params['encoding_type'] == "One-Hot Encoding":
            features = [col for col in df_encoded.columns 
                       if col not in common_params['target']]
        else:
            features = common_params['features']
        
        X = df_encoded[features]
        
        # Handle single or multiple targets
        is_multioutput = len(common_params['target']) > 1
        if is_multioutput:
            y = df[common_params['target']]
        else:
            y = df[common_params['target'][0]]
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=common_params['test_size'],
            random_state=common_params['random_seed'],
            shuffle=common_params['shuffle_data']
        )
        
        return X_train, X_test, y_train, y_test, features, is_multioutput
    
    def calculate_metrics(
        self,
        y_test,
        y_pred,
        is_multioutput: bool = False,
        target_names: List[str] = None
    ) -> None:
        """
        Calculate and display regression metrics.
        
        Args:
            y_test: True values
            y_pred: Predicted values
            is_multioutput: Whether this is multi-output regression
            target_names: Names of target variables (for multi-output)
        """
        if not is_multioutput:
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("MSE", f"{mse:.4f}")
            col2.metric("RMSE", f"{rmse:.4f}")
            col3.metric("MAE", f"{mae:.4f}")
            col4.metric("R²", f"{r2:.4f}")
        else:
            # Multi-output metrics
            for i, col in enumerate(target_names):
                st.write(f"**📊 {col}**")
                mse = mean_squared_error(y_test[col], y_pred[:, i])
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test[col], y_pred[:, i])
                r2 = r2_score(y_test[col], y_pred[:, i])
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("MSE", f"{mse:.4f}")
                c2.metric("RMSE", f"{rmse:.4f}")
                c3.metric("MAE", f"{mae:.4f}")
                c4.metric("R²", f"{r2:.4f}")
    
    def train(self, df: pd.DataFrame) -> None:
        """Main training workflow."""
        common_params = self.get_common_params(df)
        model_params = self.get_model_params()
        
        if st.button("Train Model", key=f"{self.session_key}_train_btn"):
            try:
                # Prepare data
                X_train, X_test, y_train, y_test, features, is_multioutput = \
                    self.prepare_data(df, common_params)
                
                # Create and train model
                model = self.create_model(model_params)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                # Save results
                st.session_state[self.session_key] = {
                    "model": model,
                    "X_train": X_train,
                    "X_test": X_test,
                    "y_train": y_train,
                    "y_test": y_test,
                    "y_pred": y_pred,
                    "features": features,
                    "model_params": model_params,
                    "is_multioutput": is_multioutput,
                    "target_names": common_params['target']
                }
                
                # Show success and metrics
                st.success(f"✅ {self.model_name} trained successfully!")
                self.calculate_metrics(
                    y_test, y_pred, is_multioutput, common_params['target']
                )
                
            except Exception as e:
                st.error(f"❌ Error during training: {str(e)}")
    
    def analyze(self) -> None:
        """Main analysis workflow."""
        if self.session_key not in st.session_state:
            st.warning("⚠ You must train the model first.")
            return
        
        results = st.session_state[self.session_key]
        
        st.subheader("📊 Analysis Tools")
        analysis_options = st.multiselect(
            "Select analyses to visualize",
            self.get_analysis_options(),
            default=["Actual vs Predicted"],
            key=f"{self.session_key}_analysis"
        )
        
        # Common analyses
        if "Actual vs Predicted" in analysis_options:
            self.analysis_tools.show_actual_vs_predicted(
                results["y_test"],
                results["y_pred"],
                results.get("is_multioutput", False),
                results.get("target_names"),
                key_prefix=self.session_key
            )
        
        if "Residual Plot" in analysis_options:
            self.analysis_tools.show_residual_plot(
                results["y_test"],
                results["y_pred"],
                results.get("is_multioutput", False),
                results.get("target_names"),
                key_prefix=self.session_key
            )
        
        if "Error Distribution" in analysis_options:
            self.analysis_tools.show_error_distribution(
                results["y_test"],
                results["y_pred"],
                results.get("is_multioutput", False),
                results.get("target_names"),
                key_prefix=self.session_key
            )
        
        if "Prediction Table" in analysis_options:
            self.analysis_tools.show_prediction_table(
                results["y_test"],
                results["y_pred"],
                results.get("is_multioutput", False),
                results.get("target_names")
            )
        
        if "Learning Curve" in analysis_options:
            self.analysis_tools.show_learning_curve(
                results["model"],
                results["X_train"],
                results["y_train"],
                f"Learning Curve ({self.model_name})"
            )
        
        # Model-specific analyses
        self.show_model_specific_analysis(results, analysis_options)
    
    def show_model_specific_analysis(
        self,
        results: Dict,
        options: List[str]
    ) -> None:
        """Override this in child classes for model-specific analyses."""
        pass
    
    def page(self, df: pd.DataFrame) -> None:
        """Main page function."""
        st.title(self.model_name)
        self.train(df)
        st.markdown("---")
        self.analyze()