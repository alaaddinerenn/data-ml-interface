import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Union
from sklearn.model_selection import learning_curve

from utils import DownloadManager


class RegressionAnalysisTools:
    """Utility class for regression analysis and visualization."""
    
    @staticmethod
    def _to_array(data):
        """Safely convert to numpy array."""
        if isinstance(data, np.ndarray):
            return data
        elif hasattr(data, 'values'):
            return data.values
        else:
            return np.array(data)
    
    @staticmethod
    def show_actual_vs_predicted(
        y_test,
        y_pred,
        is_multioutput: bool = False,
        target_names: Optional[List[str]] = None,
        key_prefix: str = "reg"
    ) -> None:
        """Display actual vs predicted scatter plot."""
        st.markdown("### 📈 **Actual vs Predicted Values**")
        
        if not is_multioutput:
            # Single output
            y_test_arr = RegressionAnalysisTools._to_array(y_test)
            y_pred_arr = RegressionAnalysisTools._to_array(y_pred)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(y_test_arr, y_pred_arr, alpha=0.6, edgecolors='k')
            
            # Perfect prediction line
            min_val = min(y_test_arr.min(), y_pred_arr.min())
            max_val = max(y_test_arr.max(), y_pred_arr.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
            
            ax.set_xlabel("Actual Values")
            ax.set_ylabel("Predicted Values")
            ax.set_title("Actual vs Predicted")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
            DownloadManager.download_plot(fig, "actual_vs_predicted")
            plt.close(fig)
        else:
            # Multi-output
            y_test_arr = RegressionAnalysisTools._to_array(y_test)
            y_pred_arr = RegressionAnalysisTools._to_array(y_pred)
            
            # Ensure 2D
            if y_test_arr.ndim == 1:
                y_test_arr = y_test_arr.reshape(-1, 1)
            if y_pred_arr.ndim == 1:
                y_pred_arr = y_pred_arr.reshape(-1, 1)
            
            for i, col in enumerate(target_names):
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.scatter(y_test_arr[:, i], y_pred_arr[:, i], alpha=0.6, edgecolors='k')
                
                min_val = min(y_test_arr[:, i].min(), y_pred_arr[:, i].min())
                max_val = max(y_test_arr[:, i].max(), y_pred_arr[:, i].max())
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
                
                ax.set_xlabel(f"Actual {col}")
                ax.set_ylabel(f"Predicted {col}")
                ax.set_title(f"Actual vs Predicted: {col}")
                ax.grid(True, alpha=0.3)
                
                st.pyplot(fig)
                DownloadManager.download_plot(fig, f"actual_vs_predicted_{col}")
                plt.close(fig)
    
    @staticmethod
    def show_residual_plot(
        y_test,
        y_pred,
        is_multioutput: bool = False,
        target_names: Optional[List[str]] = None,
        key_prefix: str = "reg"
    ) -> None:
        """Display residual plot."""
        st.markdown("### 📉 **Residual Plot**")
        
        if not is_multioutput:
            # Single output
            y_test_arr = RegressionAnalysisTools._to_array(y_test)
            y_pred_arr = RegressionAnalysisTools._to_array(y_pred)
            residuals = y_test_arr - y_pred_arr
            
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(y_pred_arr, residuals, alpha=0.6, edgecolors='k')
            ax.axhline(y=0, color='r', linestyle='--', lw=2)
            ax.set_xlabel("Predicted Values")
            ax.set_ylabel("Residuals")
            ax.set_title("Residual Plot")
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
            DownloadManager.download_plot(fig, "residual_plot")
            plt.close(fig)
        else:
            # Multi-output
            y_test_arr = RegressionAnalysisTools._to_array(y_test)
            y_pred_arr = RegressionAnalysisTools._to_array(y_pred)
            
            if y_test_arr.ndim == 1:
                y_test_arr = y_test_arr.reshape(-1, 1)
            if y_pred_arr.ndim == 1:
                y_pred_arr = y_pred_arr.reshape(-1, 1)
            
            for i, col in enumerate(target_names):
                residuals = y_test_arr[:, i] - y_pred_arr[:, i]
                
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.scatter(y_pred_arr[:, i], residuals, alpha=0.6, edgecolors='k')
                ax.axhline(y=0, color='r', linestyle='--', lw=2)
                ax.set_xlabel(f"Predicted {col}")
                ax.set_ylabel("Residuals")
                ax.set_title(f"Residual Plot: {col}")
                ax.grid(True, alpha=0.3)
                
                st.pyplot(fig)
                DownloadManager.download_plot(fig, f"residual_plot_{col}")
                plt.close(fig)
    
    @staticmethod
    def show_error_distribution(
        y_test,
        y_pred,
        is_multioutput: bool = False,
        target_names: Optional[List[str]] = None,
        key_prefix: str = "reg"
    ) -> None:
        """Display error distribution histogram."""
        st.markdown("### 📊 **Error Distribution**")
        
        if not is_multioutput:
            # Single output
            y_test_arr = RegressionAnalysisTools._to_array(y_test)
            y_pred_arr = RegressionAnalysisTools._to_array(y_pred)
            errors = y_test_arr - y_pred_arr
            
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.hist(errors, bins=30, edgecolor='black', alpha=0.7)
            ax.axvline(x=0, color='r', linestyle='--', lw=2, label='Zero Error')
            ax.set_xlabel("Prediction Error")
            ax.set_ylabel("Frequency")
            ax.set_title("Error Distribution")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
            DownloadManager.download_plot(fig, "error_distribution")
            plt.close(fig)
        else:
            # Multi-output
            y_test_arr = RegressionAnalysisTools._to_array(y_test)
            y_pred_arr = RegressionAnalysisTools._to_array(y_pred)
            
            if y_test_arr.ndim == 1:
                y_test_arr = y_test_arr.reshape(-1, 1)
            if y_pred_arr.ndim == 1:
                y_pred_arr = y_pred_arr.reshape(-1, 1)
            
            for i, col in enumerate(target_names):
                errors = y_test_arr[:, i] - y_pred_arr[:, i]
                
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.hist(errors, bins=30, edgecolor='black', alpha=0.7)
                ax.axvline(x=0, color='r', linestyle='--', lw=2)
                ax.set_xlabel(f"Prediction Error ({col})")
                ax.set_ylabel("Frequency")
                ax.set_title(f"Error Distribution: {col}")
                ax.grid(True, alpha=0.3)
                
                st.pyplot(fig)
                DownloadManager.download_plot(fig, f"error_distribution_{col}")
                plt.close(fig)
    
    @staticmethod
    def show_prediction_table(
        y_test,
        y_pred,
        is_multioutput: bool,
        target_names: List[str]
    ) -> None:
        """Display prediction comparison table."""
        st.markdown("### 📋 Prediction Table")
        
        # Convert to arrays
        y_test_arr = RegressionAnalysisTools._to_array(y_test)
        y_pred_arr = RegressionAnalysisTools._to_array(y_pred)
        
        if is_multioutput:
            # Ensure 2D
            if y_test_arr.ndim == 1:
                y_test_arr = y_test_arr.reshape(-1, 1)
            if y_pred_arr.ndim == 1:
                y_pred_arr = y_pred_arr.reshape(-1, 1)
            
            # Create comparison dataframe
            comparison_data = {}
            for i, target_name in enumerate(target_names):
                comparison_data[f'Actual_{target_name}'] = y_test_arr[:, i]
                comparison_data[f'Predicted_{target_name}'] = y_pred_arr[:, i]
                comparison_data[f'Error_{target_name}'] = y_test_arr[:, i] - y_pred_arr[:, i]
            
            comparison_df = pd.DataFrame(comparison_data)
        else:
            # Single output
            comparison_df = pd.DataFrame({
                'Actual': y_test_arr.ravel() if y_test_arr.ndim > 1 else y_test_arr,
                'Predicted': y_pred_arr.ravel() if y_pred_arr.ndim > 1 else y_pred_arr,
                'Error': (y_test_arr.ravel() if y_test_arr.ndim > 1 else y_test_arr) - 
                         (y_pred_arr.ravel() if y_pred_arr.ndim > 1 else y_pred_arr)
            })
        
        # Show sample
        st.write("**Sample Predictions (first 10 rows):**")
        st.dataframe(comparison_df.head(10))
        
        # Download button
        csv = comparison_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Full Prediction Table",
            data=csv,
            file_name="predictions.csv",
            mime="text/csv"
        )
    
    @staticmethod
    def show_learning_curve(model, X_train, y_train, title: str = "Learning Curve") -> None:
        """Display learning curve for regression."""
        st.markdown("### 📈 **Learning Curve**")

        train_sizes, train_scores, val_scores = learning_curve(
            estimator=model,
            X=X_train,
            y=y_train,
            cv=5,
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring="r2",
            n_jobs=-1
        )
        
        train_mean = np.mean(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(train_sizes, train_mean, 'o-', label="Training R²", linewidth=2)
        ax.plot(train_sizes, val_mean, 'o-', label="Validation R²", linewidth=2)
        ax.set_xlabel("Training Set Size")
        ax.set_ylabel("R² Score")
        ax.set_title(title)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        DownloadManager.download_plot(fig, "learning_curve")
        plt.close(fig)