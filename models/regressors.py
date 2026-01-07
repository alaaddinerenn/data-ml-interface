import streamlit as st
import pandas as pd
from typing import Dict, Any, List
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
from sklearn.multioutput import MultiOutputRegressor

from .base import BaseRegressor


# ============================================
# LINEAR REGRESSION
# ============================================

class LinearRegressionModel(BaseRegressor):
    """Linear Regression implementation."""
    
    def __init__(self):
        super().__init__("Linear Regression", "linreg_results")
    
    def get_model_params(self) -> Dict[str, Any]:
        """Linear Regression has no hyperparameters in sklearn."""
        return {}
    
    def create_model(self, params: Dict[str, Any]):
        """Create Linear Regression model."""
        return LinearRegression()
    
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
        self.scaler_X = None
        self.scaler_y = None  # ✅ ADD: Target scaler
    
    def supports_multi_output(self) -> bool:
        """SGD supports multi-output via MultiOutputRegressor."""
        return True
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get SGD specific parameters."""
        # Scaling option
        scaler_option = st.selectbox(
            "Scaling Method",
            ["StandardScaler (Z-Score)", "MinMaxScaler", "MaxAbsScaler", "None"],
            key=f"{self.session_key}_scaler"
        )
        
        # ✅ ADD: Target scaling option
        scale_target = st.checkbox(
            "Scale Target Variable (Recommended for large values)",
            value=True,
            key=f"{self.session_key}_scale_target",
            help="Scale y values - important for regression with large target values"
        )
        
        # SGD parameters
        alpha = st.number_input(
            "Alpha (Regularization)",
            min_value=0.00001,
            max_value=1.0,
            value=0.0001,
            step=0.00001,
            format="%.5f",
            key=f"{self.session_key}_alpha"
        )
        
        max_iter = st.number_input(
            "Max Iterations",
            min_value=100,
            max_value=10000,
            value=1000,
            step=100,
            key=f"{self.session_key}_max_iter"
        )
        
        # ✅ ADD: Tolerance parameter
        tol = st.number_input(
            "Tolerance (Stopping Criterion)",
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
            key=f"{self.session_key}_penalty"
        )
        
        learning_rate = st.selectbox(
            "Learning Rate Schedule",
            ["invscaling", "constant", "optimal", "adaptive"],
            index=0,  # ✅ Default to invscaling
            key=f"{self.session_key}_lr_schedule"
        )
        
        eta0 = st.number_input(
            "Eta0 (Initial Learning Rate)",
            min_value=0.0001,
            max_value=1.0,
            value=0.01,
            step=0.001,
            format="%.4f",
            key=f"{self.session_key}_eta0"
        )
        
        # ✅ ADD: Power_t for invscaling
        if learning_rate == "invscaling":
            power_t = st.number_input(
                "Power_t (for invscaling)",
                min_value=0.1,
                max_value=1.0,
                value=0.25,
                step=0.05,
                key=f"{self.session_key}_power_t"
            )
        else:
            power_t = 0.25
        
        early_stopping = st.checkbox(
            "Early Stopping",
            value=True,
            key=f"{self.session_key}_early_stop"
        )
        
        # ✅ ADD: Validation fraction for early stopping
        if early_stopping:
            validation_fraction = st.slider(
                "Validation Fraction",
                min_value=0.1,
                max_value=0.3,
                value=0.1,
                step=0.05,
                key=f"{self.session_key}_val_frac"
            )
            n_iter_no_change = st.number_input(
                "N Iter No Change",
                min_value=3,
                max_value=20,
                value=5,
                key=f"{self.session_key}_n_iter"
            )
        else:
            validation_fraction = 0.1
            n_iter_no_change = 5
        
        params = {
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
        
        return params
    
    def prepare_data(self, df, common_params):
        """Override to add scaling support for X and y."""
        X_train, X_test, y_train, y_test, features, is_multioutput = \
            super().prepare_data(df, common_params)
        
        # ✅ FIX: Read from session_state (widget key auto-stored)
        scaler_option = st.session_state.get(
            f'{self.session_key}_scaler',  # ✅ Widget key
            'StandardScaler (Z-Score)'
        )
        
        if scaler_option == "StandardScaler (Z-Score)":
            self.scaler_X = StandardScaler()
        elif scaler_option == "MinMaxScaler":
            self.scaler_X = MinMaxScaler()
        elif scaler_option == "MaxAbsScaler":
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
        
        # ✅ FIX: Read from session_state (widget key auto-stored)
        scale_target = st.session_state.get(
            f'{self.session_key}_scale_target',  # ✅ Widget key
            True
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
                    self.scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel(),
                    index=y_train.index,
                    name=y_train.name
                )
                y_test = pd.Series(
                    self.scaler_y.transform(y_test.values.reshape(-1, 1)).ravel(),
                    index=y_test.index,
                    name=y_test.name
                )
        else:
            self.scaler_y = None
        
        return X_train, X_test, y_train, y_test, features, is_multioutput
    
    def create_model(self, params: Dict[str, Any]):
        """Create SGD model."""
        return SGDRegressor(**params)
    
    def train(self, df: pd.DataFrame) -> None:
        """Override to handle multi-output and inverse scaling."""
        common_params = self.get_common_params(df)
        model_params = self.get_model_params()
        
        if st.button("Train Model", key=f"{self.session_key}_train_btn"):
            try:
                X_train, X_test, y_train, y_test, features, is_multioutput = \
                    self.prepare_data(df, common_params)
                
                # Create model
                base_model = self.create_model(model_params)
                
                # Wrap in MultiOutputRegressor if needed
                if is_multioutput:
                    model = MultiOutputRegressor(base_model)
                else:
                    model = base_model
                
                # Train
                model.fit(X_train, y_train)
                y_pred_scaled = model.predict(X_test)
                
                # ✅ FIX: Inverse transform predictions if y was scaled
                if self.scaler_y is not None:
                    if is_multioutput:
                        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
                        y_test_original = self.scaler_y.inverse_transform(y_test)
                    else:
                        y_pred = self.scaler_y.inverse_transform(
                            y_pred_scaled.reshape(-1, 1)
                        ).ravel()
                        y_test_original = self.scaler_y.inverse_transform(
                            y_test.values.reshape(-1, 1)
                        ).ravel()
                else:
                    y_pred = y_pred_scaled
                    y_test_original = y_test.values if hasattr(y_test, 'values') else y_test
                
                # Save results (with original scale for metrics)
                st.session_state[self.session_key] = {
                    "model": model,
                    "X_train": X_train,
                    "X_test": X_test,
                    "y_train": y_train,
                    "y_test": y_test_original,  # ✅ Original scale
                    "y_pred": y_pred,  # ✅ Original scale
                    "features": features,
                    "model_params": model_params,
                    "is_multioutput": is_multioutput,
                    "target_names": common_params['target'],
                    "scaler_X": self.scaler_X,
                    "scaler_y": self.scaler_y
                }
                
                st.success(f"✅ {self.model_name} trained successfully!")
                
                # ✅ Calculate metrics on original scale
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