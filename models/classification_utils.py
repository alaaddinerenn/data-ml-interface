import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Any
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import LabelEncoder
from mlxtend.plotting import plot_decision_regions

from utils import DownloadManager


class AnalysisTools:
    """
    Utility class containing common analysis and visualization methods for classifiers.
    """
    
    @staticmethod
    def show_classification_report(y_test: pd.Series, y_pred: np.ndarray) -> None:
        """
        Display classification report as a formatted DataFrame.
        
        Args:
            y_test: True labels
            y_pred: Predicted labels
        """
        report_dict = classification_report(
            y_test,
            y_pred,
            output_dict=True,
            zero_division=0
        )
        report_df = pd.DataFrame(report_dict).transpose()

        st.markdown("### **📋 Classification Report**")
        st.dataframe(report_df.style.format("{:.4f}"))
    
    @staticmethod
    def show_confusion_matrix(
        y_test: pd.Series,
        y_pred: np.ndarray,
        key_prefix: str = "cm"
    ) -> None:
        """
        Display confusion matrix with raw/normalized options.
        
        Args:
            y_test: True labels
            y_pred: Predicted labels
            key_prefix: Prefix for unique widget keys
        """
        st.markdown("### 📊 **Confusion Matrix**")

        cm_display_type = st.radio(
            "Display Type",
            ["Raw Count", "Normalized (%)"],
            horizontal=True,
            key=f"{key_prefix}_cm_type"
        )
        
        cm = confusion_matrix(y_test, y_pred)
        
        if cm_display_type == "Normalized (%)":
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
            fmt = ".2f"
        else:
            fmt = "d"
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        
        st.pyplot(fig)
        DownloadManager.download_plot(fig, f"{key_prefix}_confusion_matrix")
        plt.close(fig)
    
    @staticmethod
    def show_roc_curve(
        y_test: pd.Series,
        model: Any,
        X_test: pd.DataFrame,
        key_prefix: str = "roc"
    ) -> None:
        """
        Display ROC curve for binary or multiclass classification.
        
        Args:
            y_test: True labels
            model: Trained model with predict_proba method
            X_test: Test features
            key_prefix: Prefix for unique widget keys
        """
        st.markdown("### 📈 **ROC Curve**")
        unique_classes = sorted(list(set(y_test)))
        
        if len(unique_classes) < 2:
            st.warning("Not enough classes available for ROC Curve.")
            return
        
        # Check if model has label_encoder (for XGBoost)
        has_label_encoder = hasattr(model, 'classes_') and \
                           all(isinstance(c, (int, np.integer)) for c in model.classes_)
        
        # Binary classification
        if len(unique_classes) == 2:
            pos_label = st.selectbox(
                "Positive class (for ROC)",
                unique_classes,
                index=1,
                key=f"{key_prefix}_pos_label"
            )
            
            # Handle encoded classes
            if has_label_encoder:
                # Model classes are [0, 1], need to map from original labels
                label_to_idx = {label: idx for idx, label in enumerate(sorted(unique_classes))}
                class_idx = label_to_idx[pos_label]
            else:
                class_idx = list(model.classes_).index(pos_label)
            
            y_score = model.predict_proba(X_test)[:, class_idx]
            
            fpr, tpr, _ = roc_curve(y_test, y_score, pos_label=pos_label)
            roc_auc = auc(fpr, tpr)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(fpr, tpr, color='blue', lw=2, label=f"AUC = {roc_auc:.2f}")
            ax.plot([0, 1], [0, 1], color='gray', linestyle='--', label="Random")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title(f"ROC Curve (Positive: {pos_label})")
            ax.legend(loc="lower right")
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
            DownloadManager.download_plot(fig, f"{key_prefix}_roc_curve")
            plt.close(fig)
        
        # Multiclass classification
        else:
            selected_classes = st.multiselect(
                "Select two classes you want to compare for ROC",
                unique_classes,
                default=unique_classes[:2],
                key=f"{key_prefix}_roc_classes"
            )
            
            if len(selected_classes) != 2:
                st.warning("Please select exactly two classes.")
                return
            
            # Reset indices to avoid alignment issues
            y_test_reset = y_test.reset_index(drop=True)
            X_test_reset = X_test.reset_index(drop=True)
            
            # Filter for selected classes
            mask = y_test_reset.isin(selected_classes).to_numpy()
            y_test_bin = y_test_reset[mask].apply(
                lambda x: 1 if x == selected_classes[1] else 0
            )
            
            # Handle encoded classes for multiclass
            if has_label_encoder:
                # Model classes are [0, 1, 2, ...], need to map from original labels
                label_to_idx = {label: idx for idx, label in enumerate(sorted(unique_classes))}
                class_idx = label_to_idx[selected_classes[1]]
            else:
                class_idx = list(model.classes_).index(selected_classes[1])
            
            # Get probability scores
            y_score = model.predict_proba(X_test_reset[mask])[:, class_idx]
            
            fpr, tpr, _ = roc_curve(y_test_bin, y_score, pos_label=1)
            roc_auc = auc(fpr, tpr)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(fpr, tpr, color='blue', lw=2, label=f"AUC = {roc_auc:.2f}")
            ax.plot([0, 1], [0, 1], color='gray', linestyle='--', label="Random")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title(f"ROC Curve (Positive: {selected_classes[1]})")
            ax.legend(loc="lower right")
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
            DownloadManager.download_plot(fig, f"{key_prefix}_roc_curve")
            plt.close(fig)
    
    @staticmethod
    def show_decision_regions(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_class: type,
        model_params: dict,
        key_prefix: str = "dr"
    ) -> None:
        """
        Display decision regions for 2 selected features.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            model_class: Model class for creating new instance
            model_params: Parameters for model initialization
            key_prefix: Prefix for unique widget keys
        """
        st.markdown("### 🗺 **Decision Regions** (requires exactly 2 features)")
        
        available_features = list(X_test.columns)
        selected_features = st.multiselect(
            "Select 2 features for decision regions",
            available_features,
            default=available_features[:2] if len(available_features) >= 2 else available_features,
            key=f"{key_prefix}_decision_regions_features"
        )
        
        if len(selected_features) != 2:
            st.warning("Please select exactly two features.")
            return
        
        le = LabelEncoder()
        
        # --- Training Data ---
        X_train_sel = X_train[selected_features].to_numpy()
        y_train_sel = y_train.to_numpy()
        y_train_encoded = le.fit_transform(y_train_sel)
        
        # Create and train model for plotting
        model_for_plot = model_class(**model_params)
        model_for_plot.fit(X_train_sel, y_train_encoded)
        
        fig_train, ax_train = plt.subplots(figsize=(8, 6))
        plot_decision_regions(
            X_train_sel,
            y_train_encoded,
            clf=model_for_plot,
            legend=2,
            ax=ax_train
        )
        ax_train.set_xlabel(selected_features[0])
        ax_train.set_ylabel(selected_features[1])
        ax_train.set_title("Decision Regions - Training Data")
        
        st.pyplot(fig_train)
        DownloadManager.download_plot(fig_train, f"{key_prefix}_decision_regions_train")
        plt.close(fig_train)
        
        # --- Test Data ---
        X_test_sel = X_test[selected_features].to_numpy()
        y_test_sel = y_test.to_numpy()
        y_test_encoded = le.transform(y_test_sel)
        
        fig_test, ax_test = plt.subplots(figsize=(8, 6))
        plot_decision_regions(
            X_test_sel,
            y_test_encoded,
            clf=model_for_plot,
            legend=2,
            ax=ax_test
        )
        ax_test.set_xlabel(selected_features[0])
        ax_test.set_ylabel(selected_features[1])
        ax_test.set_title("Decision Regions - Test Data")
        
        st.pyplot(fig_test)
        DownloadManager.download_plot(fig_test, f"{key_prefix}_decision_regions_test")
        plt.close(fig_test)
    
    @staticmethod
    def show_learning_curve(
        model: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        title: str = "Learning Curve"
    ) -> None:
        """
        Display learning curve showing training and validation scores.
        
        Args:
            model: Trained model
            X_train: Training features
            y_train: Training labels
            title: Plot title
        """
        st.markdown("### 📚 **Learning Curve**")
        X = X_train.to_numpy()
        y = y_train.to_numpy()
        
        # Encode labels if needed
        if y.dtype.kind not in {'i', 'u'}:
            le = LabelEncoder()
            y = le.fit_transform(y)
        else:
            y = y.astype(int)
        
        # Calculate learning curve
        train_sizes, train_scores, val_scores = learning_curve(
            estimator=model,
            X=X,
            y=y,
            cv=5,
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring="accuracy",
            n_jobs=-1
        )
        
        # Calculate means
        train_mean = np.mean(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        
        # Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(train_sizes, train_mean, 'o-', label="Training score", linewidth=2)
        ax.plot(train_sizes, val_mean, 'o-', label="Validation score", linewidth=2)
        ax.set_xlabel("Training Set Size")
        ax.set_ylabel("Accuracy Score")
        ax.set_title(title)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        DownloadManager.download_plot(fig, "learning_curve")
        plt.close(fig)