import streamlit as st
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

from models.utils import encode_features
from models.regression_utils import RegressionAnalysisTools


class BaseRegressor(ABC):
    """
    Abstract base class for all regression models.
    
    Attributes:
        model_name (str): Display name of the model
        session_key (str): Unique key for storing results in session state
        analysis_tools (RegressionAnalysisTools): Instance of regression analysis utilities
        scaler_X (Optional): Feature scaler
        scaler_y (Optional): Target scaler
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
        self.scaler_X = None
        self.scaler_y = None
    
    @staticmethod
    def _to_array(data):
        """
        Safely convert data to numpy array.
        
        Args:
            data: Input data (numpy array, pandas Series/DataFrame, or list)
            
        Returns:
            numpy array
        """
        if isinstance(data, np.ndarray):
            return data
        elif hasattr(data, 'values'):
            return data.values
        else:
            return np.array(data)
    
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
    
    def needs_scaling(self) -> bool:
        """
        Override this if model requires/benefits from scaling.
        
        Returns:
            True if model needs feature scaling
        """
        return False
    
    def needs_target_scaling(self) -> bool:
        """
        Override this if model requires/benefits from target scaling.
        
        Returns:
            True if model needs target scaling
        """
        return False
    
    def apply_scaling(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train,
        y_test,
        features: List[str],
        is_multioutput: bool
    ) -> Tuple:
        """
        Apply scaling to features and optionally to target.
        Reads scaler options from session_state (set by widgets).
        """
        # ✅ Read scaler options from session_state
        scaler_option = st.session_state.get(
            f'{self.session_key}_scaler',
            'None'
        )
        
        # Feature scaling
        if scaler_option == "StandardScaler (Z-Score)":
            from sklearn.preprocessing import StandardScaler
            self.scaler_X = StandardScaler()
        elif scaler_option == "MinMaxScaler":
            from sklearn.preprocessing import MinMaxScaler
            self.scaler_X = MinMaxScaler()
        elif scaler_option == "MaxAbsScaler":
            from sklearn.preprocessing import MaxAbsScaler
            self.scaler_X = MaxAbsScaler()
        else:
            self.scaler_X = None
        
        if self.scaler_X:
            X_train = pd.DataFrame(
                self.scaler_X.fit_transform(X_train),
                columns=features,
                index=X_train.index
            )
            X_test = pd.DataFrame(
                self.scaler_X.transform(X_test),
                columns=features,
                index=X_test.index
            )
        
        # Target scaling
        scale_target = st.session_state.get(
            f'{self.session_key}_scale_target',
            False
        )
        
        if scale_target:
            self.scaler_y = StandardScaler()
            
            if is_multioutput:
                y_train_scaled = self.scaler_y.fit_transform(y_train)
                y_test_scaled = self.scaler_y.transform(y_test)
                
                y_train = pd.DataFrame(
                    y_train_scaled,
                    columns=y_train.columns,
                    index=y_train.index
                )
                y_test = pd.DataFrame(
                    y_test_scaled,
                    columns=y_test.columns,
                    index=y_test.index
                )
            else:
                y_train = pd.Series(
                    self.scaler_y.fit_transform(
                        y_train.values.reshape(-1, 1)
                    ).ravel(),
                    index=y_train.index,
                    name=y_train.name
                )
                y_test = pd.Series(
                    self.scaler_y.transform(
                        y_test.values.reshape(-1, 1)
                    ).ravel(),
                    index=y_test.index,
                    name=y_test.name
                )
        
        return X_train, X_test, y_train, y_test
    
    def inverse_transform_predictions(
        self,
        y_pred_scaled,
        y_test_scaled,
        is_multioutput: bool
    ) -> Tuple:
        """
        Inverse transform predictions if target was scaled.
        
        Args:
            y_pred_scaled: Scaled predictions
            y_test_scaled: Scaled test targets
            is_multioutput: Whether multi-output
            
        Returns:
            Tuple of (y_pred, y_test) in original scale
        """
        if self.scaler_y is None:
            y_pred = y_pred_scaled
            y_test = y_test_scaled.values if hasattr(y_test_scaled, 'values') else y_test_scaled
            return y_pred, y_test
        
        if is_multioutput:
            y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
            y_test = self.scaler_y.inverse_transform(y_test_scaled)
        else:
            y_pred = self.scaler_y.inverse_transform(
                y_pred_scaled.reshape(-1, 1)
            ).ravel()
            y_test = self.scaler_y.inverse_transform(
                y_test_scaled.values.reshape(-1, 1)
            ).ravel()
        
        return y_pred, y_test
    
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
        """Prepare data for training."""
        # Validate
        target_cols = common_params['target']
        if not target_cols:
            raise ValueError("You must select at least one target variable.")
        
        if not common_params['features']:
            raise ValueError("You must select at least one feature.")
        
        # Encoding
        df_encoded = encode_features(
            df,
            common_params['encoding_type'],
            target_col=target_cols
        )
        
        # Update features
        if common_params['encoding_type'] == "One-Hot Encoding":
            features = [col for col in df_encoded.columns 
                       if col not in target_cols]
        else:
            features = common_params['features']
        
        # Split
        X = df_encoded[features]
        y = df_encoded[target_cols]
        
        is_multioutput = len(target_cols) > 1
        
        if not is_multioutput:
            y = y.iloc[:, 0]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=common_params['test_size'],
            random_state=common_params['random_seed']
        )
        
        # ✅ Apply scaling (reads from session_state)
        X_train, X_test, y_train, y_test = self.apply_scaling(
            X_train, X_test, y_train, y_test,
            features, is_multioutput
        )
        
        return X_train, X_test, y_train, y_test, features, is_multioutput
    
    def calculate_metrics(
        self,
        y_test,
        y_pred,
        is_multioutput: bool,
        target_names: List[str]
    ) -> None:
        """Calculate and display regression metrics."""
        st.markdown("---")
        st.markdown("### 📊 Model Performance Metrics")
        
        # ✅ Convert to arrays
        y_test_arr = self._to_array(y_test)
        y_pred_arr = self._to_array(y_pred)
        
        if not is_multioutput:
            # Single output metrics
            mse = mean_squared_error(y_test_arr, y_pred_arr)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test_arr, y_pred_arr)
            r2 = r2_score(y_test_arr, y_pred_arr)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("MSE", f"{mse:.4f}")
            with col2:
                st.metric("RMSE", f"{rmse:.4f}")
            with col3:
                st.metric("MAE", f"{mae:.4f}")
            with col4:
                st.metric("R² Score", f"{r2:.4f}")
        else:
            # Multi-output metrics
            # Ensure 2D
            if y_test_arr.ndim == 1:
                y_test_arr = y_test_arr.reshape(-1, 1)
            if y_pred_arr.ndim == 1:
                y_pred_arr = y_pred_arr.reshape(-1, 1)
            
            for i, target_name in enumerate(target_names):
                st.markdown(f"**Target: {target_name}**")
                
                mse = mean_squared_error(y_test_arr[:, i], y_pred_arr[:, i])
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test_arr[:, i], y_pred_arr[:, i])
                r2 = r2_score(y_test_arr[:, i], y_pred_arr[:, i])
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("MSE", f"{mse:.4f}")
                with col2:
                    st.metric("RMSE", f"{rmse:.4f}")
                with col3:
                    st.metric("MAE", f"{mae:.4f}")
                with col4:
                    st.metric("R² Score", f"{r2:.4f}")
                
                st.markdown("---")
    
    def train(self, df: pd.DataFrame) -> None:
        """Main training workflow."""
        common_params = self.get_common_params(df)
        model_params = self.get_model_params()
        
        if st.button("Train Model", key=f"{self.session_key}_train_btn"):
            try:
                X_train, X_test, y_train, y_test, features, is_multioutput = \
                    self.prepare_data(df, common_params)
                
                # Create model
                model = self.create_model(model_params)
                
                # Train
                model.fit(X_train, y_train)
                y_pred_scaled = model.predict(X_test)
                
                # ✅ Inverse transform if needed
                y_pred, y_test_original = self.inverse_transform_predictions(
                    y_pred_scaled, y_test, is_multioutput
                )
                
                # Save results
                st.session_state[self.session_key] = {
                    "model": model,
                    "X_train": X_train,
                    "X_test": X_test,
                    "y_train": y_train,
                    "y_test": y_test_original,
                    "y_pred": y_pred,
                    "features": features,
                    "model_params": model_params,
                    "is_multioutput": is_multioutput,
                    "target_names": common_params['target'],
                    "scaler_X": self.scaler_X,
                    "scaler_y": self.scaler_y
                }
                
                st.success(f"✅ {self.model_name} trained successfully!")
                self.calculate_metrics(
                    y_test_original, y_pred, is_multioutput, common_params['target']
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