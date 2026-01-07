import streamlit as st
import pandas as pd
from typing import Dict, Any, List
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor

from .base import BaseRegressor


# ============================================
# LINEAR REGRESSION
# ============================================

class LinearRegressionModel(BaseRegressor):
    """Linear Regression implementation."""
    
    def __init__(self):
        super().__init__("Linear Regression", "linreg_results")
    
    def supports_multi_output(self) -> bool:
        """Linear Regression supports multi-output natively."""
        return True
    
    def needs_scaling(self) -> bool:
        """Linear Regression can benefit from scaling (optional)."""
        return True
    
    def needs_target_scaling(self) -> bool:
        """Target scaling is optional for Linear Regression."""
        return True
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get Linear Regression parameters."""
        
        # ✅ 1. SCALING OPTIONS (OPTIONAL BUT RECOMMENDED)
        st.markdown("#### 📊 Scaling Options")
        st.info("💡 Linear Regression doesn't require scaling, but it can improve numerical stability")
        
        scaler_option = st.selectbox(
            "Feature Scaling Method",
            ["None", "StandardScaler (Z-Score)", "MinMaxScaler", "MaxAbsScaler"],
            index=0,  # Default to None
            key=f"{self.session_key}_scaler",
            help="Optional: Can improve numerical stability and model interpretation"
        )
        
        scale_target = st.checkbox(
            "Scale Target Variable",
            value=False,  # Default to False for Linear Regression
            key=f"{self.session_key}_scale_target",
            help="Optional: Can improve numerical stability for large target values"
        )
        
        st.markdown("---")
        
        # ✅ 2. LINEAR REGRESSION OPTIONS
        st.markdown("#### ⚙️ Linear Regression Options")
        
        fit_intercept = st.checkbox(
            "Fit Intercept",
            value=True,
            key=f"{self.session_key}_fit_intercept",
            help="Whether to calculate the intercept (bias term) for this model"
        )
        
        st.info("ℹ️ Standard Linear Regression has no additional hyperparameters to tune")
        
        return {
            'fit_intercept': fit_intercept
        }
    
    def create_model(self, params: Dict[str, Any]):
        """Create Linear Regression model."""
        return LinearRegression(**params)
    
    def get_analysis_options(self) -> List[str]:
        """Get available analysis options."""
        return [
            "Actual vs Predicted",
            "Residual Plot",
            "Error Distribution",
            "Prediction Table"
        ]


# ============================================
# SGD REGRESSOR
# ============================================

class SGDRegressorModel(BaseRegressor):
    """SGD Regressor implementation with scaling support."""
    
    def __init__(self):
        super().__init__("SGD Regressor", "sgd_results")
    
    def supports_multi_output(self) -> bool:
        return True
    
    def needs_scaling(self) -> bool:
        return True
    
    def needs_target_scaling(self) -> bool:
        return True
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get SGD specific parameters."""
        
        # ✅ 1. SCALING OPTIONS FIRST (BEFORE TRAIN BUTTON)
        st.markdown("#### 📊 Scaling Options")
        st.info("⚡ SGD requires feature scaling for optimal performance")
        
        scaler_option = st.selectbox(
            "Feature Scaling Method",
            ["StandardScaler (Z-Score)", "MinMaxScaler", "MaxAbsScaler", "None"],
            index=0,  # Default to StandardScaler
            key=f"{self.session_key}_scaler",
            help="Standardizes features by removing mean and scaling to unit variance"
        )
        
        scale_target = st.checkbox(
            "Scale Target Variable",
            value=True,
            key=f"{self.session_key}_scale_target",
            help="Recommended when target values are large (e.g., California Housing prices)"
        )
        
        # ✅ 2. SGD HYPERPARAMETERS
        st.markdown("#### ⚙️ SGD Hyperparameters")
        
        alpha = st.number_input(
            "Alpha (Regularization Strength)",
            min_value=0.00001,
            max_value=1.0,
            value=0.0001,
            step=0.00001,
            format="%.5f",
            key=f"{self.session_key}_alpha",
            help="Constant that multiplies the regularization term"
        )
        
        max_iter = st.number_input(
            "Max Iterations",
            min_value=100,
            max_value=10000,
            value=1000,
            step=100,
            key=f"{self.session_key}_max_iter",
            help="Maximum number of passes over training data"
        )
        
        tol = st.number_input(
            "Tolerance (Convergence Criterion)",
            min_value=1e-5,
            max_value=1e-2,
            value=1e-3,
            step=1e-4,
            format="%.5f",
            key=f"{self.session_key}_tol",
            help="Training stops when loss > best_loss - tol for n_iter_no_change epochs"
        )
        
        penalty_choice = st.selectbox(
            "Penalty Type",
            ["l2", "l1", "elasticnet", "None"],
            index=0,
            key=f"{self.session_key}_penalty",
            help="Regularization type: l2 (Ridge), l1 (Lasso), elasticnet, or None"
        )
        
        learning_rate = st.selectbox(
            "Learning Rate Schedule",
            ["invscaling", "constant", "optimal", "adaptive"],
            index=0,
            key=f"{self.session_key}_lr_schedule",
            help="Learning rate schedule for weight updates"
        )
        
        eta0 = st.number_input(
            "Eta0 (Initial Learning Rate)",
            min_value=0.0001,
            max_value=1.0,
            value=0.01,
            step=0.001,
            format="%.4f",
            key=f"{self.session_key}_eta0",
            help="Initial learning rate for constant, invscaling or adaptive schedules"
        )
        
        if learning_rate == "invscaling":
            power_t = st.number_input(
                "Power_t (Exponent for Inverse Scaling)",
                min_value=0.1,
                max_value=1.0,
                value=0.25,
                step=0.05,
                key=f"{self.session_key}_power_t",
                help="Exponent for inverse scaling learning rate"
            )
        else:
            power_t = 0.25
        
        early_stopping = st.checkbox(
            "Early Stopping",
            value=True,
            key=f"{self.session_key}_early_stop",
            help="Stop training when validation score is not improving"
        )
        
        if early_stopping:
            validation_fraction = st.slider(
                "Validation Fraction",
                min_value=0.1,
                max_value=0.3,
                value=0.1,
                step=0.05,
                key=f"{self.session_key}_val_frac",
                help="Proportion of training data to set aside for validation"
            )
            n_iter_no_change = st.number_input(
                "N Iter No Change",
                min_value=3,
                max_value=20,
                value=5,
                key=f"{self.session_key}_n_iter",
                help="Number of iterations with no improvement to wait before stopping"
            )
        else:
            validation_fraction = 0.1
            n_iter_no_change = 5
        
        return {
            'alpha': alpha,
            'max_iter': int(max_iter),
            'tol': tol,
            'penalty': None if penalty_choice == "None" else penalty_choice,
            'learning_rate': learning_rate,
            'eta0': eta0,
            'power_t': power_t,
            'early_stopping': early_stopping,
            'validation_fraction': validation_fraction,
            'n_iter_no_change': int(n_iter_no_change),
            'random_state': 42,
            'verbose': 0
        }
    
    def create_model(self, params: Dict[str, Any]):
        """Create SGD model."""
        return SGDRegressor(**params)
    
    def train(self, df: pd.DataFrame) -> None:
        """Override to handle multi-output."""
        common_params = self.get_common_params(df)
        model_params = self.get_model_params()
        
        if st.button("Train Model", key=f"{self.session_key}_train_btn"):
            try:
                X_train, X_test, y_train, y_test, features, is_multioutput = \
                    self.prepare_data(df, common_params)
                
                # Create model
                base_model = self.create_model(model_params)
                
                # Wrap if multi-output
                if is_multioutput:
                    model = MultiOutputRegressor(base_model)
                else:
                    model = base_model
                
                # Train
                model.fit(X_train, y_train)
                y_pred_scaled = model.predict(X_test)
                
                # Inverse transform
                y_pred, y_test_original = self.inverse_transform_predictions(
                    y_pred_scaled, y_test, is_multioutput
                )
                
                # Save
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
                st.error(f"❌ Error: {str(e)}")
                import traceback
                st.error(traceback.format_exc())
    
    def get_analysis_options(self) -> List[str]:
        """Get available analysis options."""
        return [
            "Actual vs Predicted",
            "Residual Plot",
            "Error Distribution",
            "Prediction Table",
            "Learning Curve"
        ]


# ============================================
# KNN REGRESSOR
# ============================================

class KNNRegressorModel(BaseRegressor):
    """K-Nearest Neighbors Regressor with optional scaling."""
    
    def __init__(self):
        super().__init__("KNN Regressor", "knn_reg_results")
    
    def supports_multi_output(self) -> bool:
        """KNN supports multi-output regression natively."""
        return True
    
    def needs_scaling(self) -> bool:
        return True
    
    def needs_target_scaling(self) -> bool:
        return True
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get KNN specific parameters."""
        
        # ✅ 1. SCALING OPTIONS FIRST
        st.markdown("#### 📊 Scaling Options")
        st.info("🎯 KNN is distance-based and strongly recommended to use feature scaling")
        
        scaler_option = st.selectbox(
            "Feature Scaling Method",
            ["StandardScaler (Z-Score)", "MinMaxScaler", "MaxAbsScaler", "None"],  # ✅ Added None
            index=0,
            key=f"{self.session_key}_scaler",
            help="Distance-based algorithms require all features to be on the same scale"
        )
        
        scale_target = st.checkbox(
            "Scale Target Variable",
            value=False,
            key=f"{self.session_key}_scale_target",
            help="Can improve performance for large target values"
        )
        
        # ✅ 2. KNN HYPERPARAMETERS
        st.markdown("#### ⚙️ KNN Hyperparameters")
        
        n_neighbors = st.slider(
            "K Value (Number of Neighbors)",
            min_value=1,
            max_value=20,
            value=5,
            key=f"{self.session_key}_k",
            help="Number of neighbors to use for prediction"
        )
        
        metric = st.selectbox(
            "Distance Metric",
            ["euclidean", "manhattan", "minkowski"],
            index=0,
            key=f"{self.session_key}_metric",
            help="Distance metric: euclidean (L2), manhattan (L1), or minkowski"
        )
        
        weights = st.selectbox(
            "Weight Function",
            ["uniform", "distance"],
            index=0,
            key=f"{self.session_key}_weights",
            help="uniform: all neighbors weighted equally | distance: closer neighbors have more influence"
        )
        
        return {
            'n_neighbors': n_neighbors,
            'metric': metric,
            'weights': weights
        }
    
    def create_model(self, params: Dict[str, Any]):
        """Create KNN model."""
        return KNeighborsRegressor(**params)
    
    def get_analysis_options(self) -> List[str]:
        """Get available analysis options."""
        return [
            "Actual vs Predicted",
            "Residual Plot",
            "Error Distribution",
            "Prediction Table"
        ]


# ============================================
# PUBLIC API
# ============================================

def linear_regression_page(df: pd.DataFrame) -> None:
    """Entry point for Linear Regression page."""
    model = LinearRegressionModel()
    model.page(df)


def sgd_regression_page(df: pd.DataFrame) -> None:
    """Entry point for SGD Regressor page."""
    model = SGDRegressorModel()
    model.page(df)


def knn_regression_page(df: pd.DataFrame) -> None:
    """Entry point for KNN Regressor page."""
    model = KNNRegressorModel()
    model.page(df)