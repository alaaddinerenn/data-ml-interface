import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    MaxAbsScaler,
    LabelEncoder
)
from sklearn.model_selection import GridSearchCV
import xgboost as xgb

from .base import BaseClassifier
from utils import DownloadManager


# ============================================
# DECISION TREE CLASSIFIER
# ============================================

class DecisionTreeModel(BaseClassifier):
    """Decision Tree Classifier implementation."""
    
    def __init__(self):
        super().__init__("Decision Tree Classifier", "dt_results")
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get Decision Tree specific parameters."""
        max_depth = st.slider(
            "Maximum Depth",
            1, 20, 5,
            key=f"{self.session_key}_max_depth"
        )
        
        criterion = st.selectbox(
            "Splitting Criterion",
            ["gini", "entropy", "log_loss"],
            key=f"{self.session_key}_criterion"
        )
        
        return {
            'max_depth': max_depth,
            'criterion': criterion,
            'random_state': 42
        }
    
    def create_model(self, params: Dict[str, Any]):
        """Create Decision Tree model."""
        return DecisionTreeClassifier(**params)
    
    def get_analysis_options(self) -> List[str]:
        """Get available analysis options."""
        return [
            "Classification Report",
            "Confusion Matrix",
            "ROC Curve",
            "Decision Tree Plot",
            "Decision Regions",
            "Learning Curve"
        ]
    
    def show_model_specific_analysis(
        self,
        results: Dict,
        options: List[str]
    ) -> None:
        """Show Decision Tree specific visualizations."""
        if "Decision Tree Plot" in options:
            st.write("🌳 **Decision Tree Visualization**")
            
            fig, ax = plt.subplots(figsize=(15, 8))
            plot_tree(
                results["model"],
                feature_names=results["X_train"].columns,
                class_names=[str(c) for c in sorted(set(results["y_test"]))],
                filled=True,
                ax=ax,
                fontsize=10,
                rounded=True
            )
            
            st.pyplot(fig)
            DownloadManager.download_plot(fig, "decision_tree")
            plt.close(fig)


# ============================================
# KNN CLASSIFIER
# ============================================

class KNNModel(BaseClassifier):
    """K-Nearest Neighbors Classifier implementation."""
    
    def __init__(self):
        super().__init__("KNN Classifier", "knn_results")
        self.scaler = None
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get KNN specific parameters."""
        # Scaling option
        scaler_option = st.selectbox(
            "Scaling Method",
            ["StandardScaler (Z-Score)", "MinMaxScaler", "MaxAbsScaler", "None"],
            key=f"{self.session_key}_scaler"
        )
        
        # Parameter selection mode
        param_mode = st.radio(
            "Parameter Selection",
            ["Manual", "Automatic (GridSearchCV)"],
            key=f"{self.session_key}_param_mode"
        )
        
        params = {
            'scaler_option': scaler_option,
            'param_mode': param_mode
        }
        
        if param_mode == "Manual":
            params['n_neighbors'] = st.slider(
                "K Value (Number of Neighbors)",
                1, 20, 5,
                key=f"{self.session_key}_k"
            )
            params['metric'] = st.selectbox(
                "Distance Metric",
                ["euclidean", "minkowski", "manhattan", "chebyshev"],
                key=f"{self.session_key}_metric"
            )
        
        return params
    
    def prepare_data(self, df, common_params):
        """Override to add scaling support."""
        X_train, X_test, y_train, y_test, features = super().prepare_data(
            df, common_params
        )
        
        # Apply scaling if selected
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
        
        return X_train, X_test, y_train, y_test, features
    
    def create_model(self, params: Dict[str, Any]):
        """Create KNN model."""
        # Store scaler option in session state
        st.session_state[f'{self.session_key}_scaler_option'] = params.pop('scaler_option')
        param_mode = params.pop('param_mode')
        
        if param_mode == "Automatic (GridSearchCV)":
            return None  # Will be handled in train method
        
        return KNeighborsClassifier(**params)
    
    def train(self, df: pd.DataFrame) -> None:
        """Override to handle GridSearchCV."""
        common_params = self.get_common_params(df)
        model_params = self.get_model_params()
        
        if st.button("Train Model", key=f"{self.session_key}_train_btn"):
            try:
                X_train, X_test, y_train, y_test, features = self.prepare_data(
                    df, common_params
                )
                
                # Handle GridSearchCV
                if model_params['param_mode'] == "Automatic (GridSearchCV)":
                    st.info("🔍 Running GridSearchCV...")
                    
                    param_grid = {
                        "n_neighbors": list(range(1, 21)),
                        "metric": ["euclidean", "minkowski", "manhattan", "chebyshev"]
                    }
                    
                    grid = GridSearchCV(
                        KNeighborsClassifier(),
                        param_grid,
                        cv=5,
                        scoring="accuracy",
                        n_jobs=-1
                    )
                    grid.fit(X_train, y_train)
                    
                    # Find best parameters
                    results_df = pd.DataFrame(grid.cv_results_)
                    best_score = results_df['mean_test_score'].max()
                    best_rows = results_df[
                        results_df['mean_test_score'].between(
                            best_score - 0.0001,
                            best_score + 0.0001
                        )
                    ]
                    
                    best_k = best_rows['param_n_neighbors'].min()
                    best_metric = sorted(
                        best_rows[best_rows['param_n_neighbors'] == best_k]['param_metric']
                    )[0]
                    
                    model = KNeighborsClassifier(
                        n_neighbors=best_k,
                        metric=best_metric
                    )
                    st.success(f"✨ Best parameters: k={best_k}, metric={best_metric}")
                else:
                    model = self.create_model(model_params)
                
                # Train and evaluate
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                from sklearn.metrics import accuracy_score
                accuracy = accuracy_score(y_test, y_pred)
                
                # Save results
                st.session_state[self.session_key] = {
                    "model": model,
                    "X_test": X_test,
                    "X_train": X_train,
                    "y_test": y_test,
                    "y_train": y_train,
                    "y_pred": y_pred,
                    "features": features,
                    "model_params": model.get_params(),
                    "scaler": self.scaler
                }
                
                st.success(f"✅ {self.model_name} trained successfully!")
                st.write(f"**Accuracy:** {accuracy:.4f}")
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    def get_analysis_options(self) -> List[str]:
        """Get available analysis options."""
        return [
            "Classification Report",
            "Confusion Matrix",
            "ROC Curve",
            "Decision Regions",
            "Learning Curve"
        ]


# ============================================
# RANDOM FOREST CLASSIFIER
# ============================================

class RandomForestModel(BaseClassifier):
    """Random Forest Classifier implementation."""
    
    def __init__(self):
        super().__init__("Random Forest Classifier", "rf_results")
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get Random Forest specific parameters."""
        n_estimators = st.slider(
            "Number of Trees (n_estimators)",
            10, 200, 100, 10,
            key=f"{self.session_key}_n_estimators"
        )
        
        max_depth = st.slider(
            "Maximum Depth",
            1, 20, 5,
            key=f"{self.session_key}_max_depth"
        )
        
        criterion = st.selectbox(
            "Splitting Criterion",
            ["gini", "entropy", "log_loss"],
            key=f"{self.session_key}_criterion"
        )
        
        return {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'criterion': criterion,
            'random_state': 42,
            'n_jobs': -1
        }
    
    def create_model(self, params: Dict[str, Any]):
        """Create Random Forest model."""
        return RandomForestClassifier(**params)
    
    def get_analysis_options(self) -> List[str]:
        """Get available analysis options."""
        return [
            "Classification Report",
            "Confusion Matrix",
            "ROC Curve",
            "Feature Importance",
            "Decision Regions",
            "Learning Curve"
        ]
    
    def show_model_specific_analysis(
        self,
        results: Dict,
        options: List[str]
    ) -> None:
        """Show Random Forest specific visualizations."""
        if "Feature Importance" in options:
            st.markdown("### 📊 **Feature Importance**")

            importances = results["model"].feature_importances_
            fi_df = pd.DataFrame({
                "Feature": results["features"],
                "Importance": importances
            }).sort_values("Importance", ascending=False)
            
            # Plot
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(
                x="Importance",
                y="Feature",
                data=fi_df,
                ax=ax,
                hue="Feature",
                legend=False,
                palette="viridis"
            )
            ax.set_title("Feature Importances (Random Forest)")
            
            st.pyplot(fig)
            DownloadManager.download_plot(fig, "feature_importance")
            plt.close(fig)
            
            # DataFrame
            st.dataframe(fi_df)


# ============================================
# XGBOOST CLASSIFIER
# ============================================

class XGBoostModel(BaseClassifier):
    """XGBoost Classifier implementation."""
    
    def __init__(self):
        super().__init__("XGBoost Classifier", "xgb_results")
        self.label_encoder = None
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get XGBoost specific parameters."""
        n_estimators = st.slider(
            "Number of Trees (n_estimators)",
            10, 200, 100, 10,
            key=f"{self.session_key}_n_estimators"
        )
        
        max_depth = st.slider(
            "Maximum Depth",
            1, 20, 5,
            key=f"{self.session_key}_max_depth"
        )
        
        learning_rate = st.slider(
            "Learning Rate",
            0.01, 0.5, 0.1, 0.01,
            key=f"{self.session_key}_lr"
        )
        
        return {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'random_state': 42,
            'eval_metric': 'mlogloss',
            'n_jobs': -1
        }
    
    def prepare_data(self, df, common_params):
        """Override to add label encoding for target."""
        X_train, X_test, y_train, y_test, features = super().prepare_data(
            df, common_params
        )
        
        # Store original labels before encoding
        original_y_train = y_train.copy()
        original_y_test = y_test.copy()
        
        # XGBoost requires integer labels
        self.label_encoder = LabelEncoder()
        y_train_encoded = pd.Series(
            self.label_encoder.fit_transform(y_train),
            index=y_train.index,
            name=y_train.name
        )
        y_test_encoded = pd.Series(
            self.label_encoder.transform(y_test),
            index=y_test.index,
            name=y_test.name
        )
        
        # Return both encoded and original labels
        return X_train, X_test, y_train_encoded, y_test_encoded, features, original_y_train, original_y_test
    
    def create_model(self, params: Dict[str, Any]):
        """Create XGBoost model."""
        return xgb.XGBClassifier(**params)
    
    def train(self, df: pd.DataFrame) -> None:
        """Override to store both encoded and original labels."""
        common_params = self.get_common_params(df)
        model_params = self.get_model_params()
        
        if st.button("Train Model", key=f"{self.session_key}_train_btn"):
            try:
                # Unpack original labels too
                X_train, X_test, y_train, y_test, features, original_y_train, original_y_test = self.prepare_data(
                    df, common_params
                )
                
                model = self.create_model(model_params)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                # Decode predictions for accuracy calculation
                y_pred_decoded = pd.Series(
                    self.label_encoder.inverse_transform(y_pred),
                    index=y_test.index
                )
                
                from sklearn.metrics import accuracy_score
                accuracy = accuracy_score(original_y_test, y_pred_decoded)
                
                # Save ORIGINAL (decoded) labels for analysis
                st.session_state[self.session_key] = {
                    "model": model,
                    "X_test": X_test,
                    "X_train": X_train,
                    "y_test": original_y_test,  # Original labels
                    "y_train": original_y_train,  # Original labels
                    "y_pred": y_pred_decoded,  # Decoded predictions
                    "features": features,
                    "model_params": model_params,
                    "label_encoder": self.label_encoder,
                    # Keep encoded versions for model operations
                    "y_test_encoded": y_test,
                    "y_pred_encoded": y_pred
                }
                
                st.success(f"✅ {self.model_name} trained successfully!")
                st.write(f"**Accuracy:** {accuracy:.4f}")
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                import traceback
                st.error(traceback.format_exc())
    
    def get_analysis_options(self) -> List[str]:
        """Get available analysis options."""
        return [
            "Classification Report",
            "Confusion Matrix",
            "ROC Curve",
            "Feature Importance",
            "Decision Regions",
            "Learning Curve"
        ]
    
    def analyze(self) -> None:
        """Override - no need to decode, already decoded in train()."""
        if self.session_key not in st.session_state:
            st.warning("⚠ You must train the model first.")
            return
        
        # Just call parent analyze, labels are already decoded
        super().analyze()
    
    def show_model_specific_analysis(
        self,
        results: Dict,
        options: List[str]
    ) -> None:
        """Show XGBoost specific visualizations."""
        if "Feature Importance" in options:
            st.markdown("### 📊 **Feature Importance (XGBoost)**")

            booster = results["model"].get_booster()
            
            for importance_type in ["weight", "gain", "cover"]:
                importances = booster.get_score(importance_type=importance_type)
                
                if not importances:
                    st.warning(f"No importance scores available for {importance_type}")
                    continue
                
                fi_df = pd.DataFrame(
                    list(importances.items()),
                    columns=["Feature", "Importance"]
                ).sort_values("Importance", ascending=False)
                
                # Plot
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(
                    x="Importance",
                    y="Feature",
                    data=fi_df,
                    ax=ax,
                    hue="Feature",
                    legend=False,
                    palette="rocket"
                )
                ax.set_title(f"Feature Importances ({importance_type.upper()})")
                
                st.pyplot(fig)
                DownloadManager.download_plot(fig, f"xgb_feature_importance_{importance_type}")
                plt.close(fig)
                
                # DataFrame
                with st.expander(f"📋 {importance_type.upper()} values"):
                    st.dataframe(fi_df)


# ============================================
# PUBLIC API FUNCTIONS
# ============================================

def decision_tree_page(df: pd.DataFrame) -> None:
    """Entry point for Decision Tree page."""
    model = DecisionTreeModel()
    model.page(df)


def knn_page(df: pd.DataFrame) -> None:
    """Entry point for KNN page."""
    model = KNNModel()
    model.page(df)


def random_forest_page(df: pd.DataFrame) -> None:
    """Entry point for Random Forest page."""
    model = RandomForestModel()
    model.page(df)


def xgboost_classifier_page(df: pd.DataFrame) -> None:
    """Entry point for XGBoost page."""
    model = XGBoostModel()
    model.page(df)