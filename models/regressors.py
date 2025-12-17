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
        self.scaler = None
    
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
        
        # SGD parameters
        alpha = st.number_input(
            "Alpha (Regularization)",
            min_value=0.0001,
            value=0.0001,
            step=0.0001,
            format="%.4f",
            key=f"{self.session_key}_alpha"
        )
        
        max_iter = st.number_input(
            "Max Iterations",
            min_value=100,
            value=1000,
            step=100,
            key=f"{self.session_key}_max_iter"
        )
        
        penalty_choice = st.selectbox(
            "Penalty Type",
            ["l2", "l1", "elasticnet", "None"],
            key=f"{self.session_key}_penalty"
        )
        
        learning_rate = st.selectbox(
            "Learning Rate Schedule",
            ["constant", "optimal", "invscaling", "adaptive"],
            key=f"{self.session_key}_lr_schedule"
        )
        
        eta0 = st.number_input(
            "Eta0 (Initial Learning Rate)",
            min_value=0.00001,
            value=0.01,
            step=0.001,
            format="%.5f",
            key=f"{self.session_key}_eta0"
        )
        
        early_stopping = st.checkbox(
            "Early Stopping",
            value=True,
            key=f"{self.session_key}_early_stop"
        )
        
        return {
            'scaler_option': scaler_option,
            'alpha': alpha,
            'max_iter': int(max_iter),
            'penalty': None if penalty_choice == "None" else penalty_choice,
            'learning_rate': learning_rate,
            'eta0': eta0,
            'early_stopping': early_stopping,
            'random_state': 42
        }
    
    def prepare_data(self, df, common_params):
        """Override to add scaling support."""
        X_train, X_test, y_train, y_test, features, is_multioutput = \
            super().prepare_data(df, common_params)
        
        # Apply scaling
        scaler_option = st.session_state.get(
            f'{self.session_key}_scaler_option',
            'None'
        )
        
        if scaler_option == "StandardScaler (Z-Score)":
            self.scaler = StandardScaler()
        elif scaler_option == "MinMaxScaler":
            self.scaler = MinMaxScaler()
        elif scaler_option == "MaxAbsScaler":
            self.scaler = MaxAbsScaler()
        else:
            self.scaler = None
        
        if self.scaler:
            X_train = pd.DataFrame(
                self.scaler.fit_transform(X_train),
                columns=features,
                index=X_train.index
            )
            X_test = pd.DataFrame(
                self.scaler.transform(X_test),
                columns=features,
                index=X_test.index
            )
        
        return X_train, X_test, y_train, y_test, features, is_multioutput
    
    def create_model(self, params: Dict[str, Any]):
        """Create SGD model."""
        # Store scaler option
        st.session_state[f'{self.session_key}_scaler_option'] = params.pop('scaler_option')
        
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
                
                # Wrap in MultiOutputRegressor if needed
                if is_multioutput:
                    model = MultiOutputRegressor(base_model)
                else:
                    model = base_model
                
                # Train
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
                    "target_names": common_params['target'],
                    "scaler": self.scaler
                }
                
                st.success(f"✅ {self.model_name} trained successfully!")
                self.calculate_metrics(
                    y_test, y_pred, is_multioutput, common_params['target']
                )
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
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