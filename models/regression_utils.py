import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List
from sklearn.model_selection import learning_curve

from utils import DownloadManager


class RegressionAnalysisTools:
    """Common analysis and visualization tools for regressors."""
    
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
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(y_test, y_pred, alpha=0.6, edgecolors='k')
            
            # Perfect prediction line
            min_val = min(y_test.min(), y_pred.min())
            max_val = max(y_test.max(), y_pred.max())
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
            # Multi-output: separate plot for each target
            for i, col in enumerate(target_names):
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.scatter(y_test[col], y_pred[:, i], alpha=0.6, edgecolors='k')
                
                min_val = min(y_test[col].min(), y_pred[:, i].min())
                max_val = max(y_test[col].max(), y_pred[:, i].max())
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
            residuals = y_test - y_pred
            
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(y_pred, residuals, alpha=0.6, edgecolors='k')
            ax.axhline(y=0, color='r', linestyle='--', lw=2)
            ax.set_xlabel("Predicted Values")
            ax.set_ylabel("Residuals")
            ax.set_title("Residual Plot")
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
            DownloadManager.download_plot(fig, "residual_plot")
            plt.close(fig)
        else:
            for i, col in enumerate(target_names):
                residuals = y_test[col] - y_pred[:, i]
                
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.scatter(y_pred[:, i], residuals, alpha=0.6, edgecolors='k')
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
            errors = y_test - y_pred
            
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
            for i, col in enumerate(target_names):
                errors = y_test[col] - y_pred[:, i]
                
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.hist(errors, bins=30, edgecolor='black', alpha=0.7)
                ax.axvline(x=0, color='r', linestyle='--', lw=2)
                ax.set_xlabel(f"Prediction Error ({col})")
                ax.set_ylabel("Frequency")
                ax.set_title(f"Error Distribution: {col}")
                ax.grid(True, alpha=0.3)
                
                st.pyplot(fig)
                download_plot(fig, f"error_distribution_{col}")
                plt.close(fig)
    
    @staticmethod
    def show_prediction_table(
        y_test,
        y_pred,
        is_multioutput: bool = False,
        target_names: Optional[List[str]] = None
    ) -> None:
        """Display prediction table."""
        st.markdown("### 📄 **Prediction Table**")
        
        if not is_multioutput:
            combined = pd.DataFrame({
                'Actual': y_test.values,
                'Predicted': y_pred,
                'Error': y_test.values - y_pred
            })
            st.dataframe(combined)
        else:
            pred_df = pd.DataFrame(
                y_pred,
                columns=[f"Predicted_{col}" for col in target_names]
            )
            combined = pd.concat([
                y_test.reset_index(drop=True),
                pred_df
            ], axis=1)
            
            # Add error columns
            for i, col in enumerate(target_names):
                combined[f"Error_{col}"] = combined[col] - combined[f"Predicted_{col}"]
            
            st.dataframe(combined)
    
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