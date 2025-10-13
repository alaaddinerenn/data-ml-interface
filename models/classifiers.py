import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.plotting import plot_decision_regions
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, mean_squared_error, r2_score
from utils import download_plot
from models.utils import encode_features
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb


def train_decision_tree(df) -> None:
    st.subheader("🔹 Model Settings")

    # Select target column
    target = st.selectbox(
        "Select Target Column", 
        options=df.columns, 
        index=list(df.columns).index('label') if 'label' in df.columns else 0
    )

    # Select features
    features = st.multiselect(
        "Select Features for the Model", 
        options=[col for col in df.columns if col != target],
        default=[col for col in df.columns if col != target]
    )

    # Select encoding type
    encoding_type = st.radio("Encoding Type", ["One-Hot Encoding", "Label Encoding"])

    # Train-test settings
    test_size = st.slider("Test Set Ratio", 0.1, 0.5, 0.2, 0.05)
    random_seed = st.number_input("Random Seed", value=42)
    shuffle_data = st.checkbox("Shuffle Data", value=True)
    use_stratify = st.checkbox("Stratify (Preserve Class Distribution)", value=True)

    # Decision Tree parameters
    max_depth = st.slider("Maximum Depth", 1, 20, 5)
    criterion = st.selectbox("Splitting Criterion", ["gini", "entropy", "log_loss"])

    if st.button("Train Model"):
        if not features:
            st.warning("You must select at least one feature.")
            return

        # Encoding
        df_encoded = encode_features(df, encoding_type, target_col=target)
        X = df_encoded[features]
        y = df[target]

        # Stratify setting
        stratify_param = y if use_stratify else None

        # Split into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_seed, shuffle=shuffle_data, stratify=stratify_param
        )

        # Model
        model = DecisionTreeClassifier(
            criterion=criterion,
            max_depth=max_depth,
            random_state=random_seed
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Determine feature types
        feature_types = {}
        unique_values = {}
        for feature in features:
            if df[feature].dtype == "object" or df[feature].nunique() < 10:  # Categorical feature
                feature_types[feature] = "categorical"
                unique_values[feature] = df[feature].unique().tolist()
            else:  # Numerical feature
                feature_types[feature] = "numerical"

        # Save to session
        st.session_state["dt_results"] = {
            "model": model,
            "X_test": X_test,
            "X_train": X_train,
            "y_test": y_test, 
            "y_train": y_train,
            "y_pred": y_pred,
            "features": features,
            "feature_types": feature_types,
            "unique_values": unique_values
        }
        

def decision_tree_analysis() -> None:
    if "dt_results" not in st.session_state:
        st.warning("⚠ You must train the model first.")
        return

    results = st.session_state["dt_results"]
    y_test = results["y_test"]
    y_pred = results["y_pred"]

    st.success("✅ Decision Tree model trained successfully!")
    st.write(f"**Accuracy:** {accuracy_score(y_test, y_pred):.4f}")

    st.subheader("📊 Analysis Tools")
    analysis_options = st.multiselect(
        "Select analyses you want to visualize",
        ["Classification Report", "Confusion Matrix", "ROC Curve", "Decision Tree Plot", "Decision Regions", "Learning Curve"],
        default=["Classification Report"]
    )

    if "Classification Report" in analysis_options:
        report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True, zero_division=0)).transpose()
        st.dataframe(report_df)

    if "Confusion Matrix" in analysis_options:
        st.write("📊 **Confusion Matrix**")
        cm_display_type = st.radio("Display Type", ["Raw Count", "Normalized (%)"], horizontal=True)

        cm = confusion_matrix(y_test, y_pred)

        if cm_display_type == "Normalized (%)":
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
            fmt = ".2f"
        else:
            fmt = "d"
        
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt=fmt, cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
        download_plot(fig, "confusion_matrix")

    # --- ROC Curve ---
    if "ROC Curve" in analysis_options:
        unique_classes = sorted(list(set(y_test)))

        if len(unique_classes) == 2:
            pos_label = st.selectbox("Positive class (for ROC)", unique_classes, index=1)
            fpr, tpr, _ = roc_curve(
                y_test, 
                results["model"].predict_proba(results["X_test"])[:, list(results["model"].classes_).index(pos_label)], 
                pos_label=pos_label
            )
            roc_auc = auc(fpr, tpr)
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, color='blue', lw=2, label=f"AUC = {roc_auc:.2f}")
            ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title(f"ROC Curve (Positive: {pos_label})")
            ax.legend(loc="lower right")
            st.pyplot(fig)
            download_plot(fig, "roc_curve")

        elif len(unique_classes) > 2:
            selected_classes = st.multiselect(
                "Select two classes you want to compare for ROC",
                unique_classes,
                default=unique_classes[:2]
            )
            if len(selected_classes) == 2:
                mask = y_test.isin(selected_classes)
                y_test_bin = y_test[mask].apply(lambda x: 1 if x == selected_classes[1] else 0)
                y_score = results["model"].predict_proba(results["X_test"][mask])[:, list(results["model"].classes_).index(selected_classes[1])]
                fpr, tpr, _ = roc_curve(y_test_bin, y_score, pos_label=1)
                roc_auc = auc(fpr, tpr)
                fig, ax = plt.subplots()
                ax.plot(fpr, tpr, color='blue', lw=2, label=f"AUC = {roc_auc:.2f}")
                ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")
                ax.set_title(f"ROC Curve (Positive: {selected_classes[1]})")
                ax.legend(loc="lower right")
                st.pyplot(fig)
                download_plot(fig, "roc_curve")
            else:
                st.warning("Please select exactly two classes.")
        else:
            st.warning("Not enough classes available for ROC Curve.")

    if "Decision Tree Plot" in analysis_options:
        fig, ax = plt.subplots(figsize=(12, 6))
        plot_tree(results["model"], feature_names=results["X_train"].columns, class_names=[str(c) for c in set(y_test)], filled=True, ax=ax, fontsize=12, rounded=True)
        st.pyplot(fig)
        download_plot(fig, "tree")

    if "Decision Regions" in analysis_options:
        available_features = list(results["X_test"].columns)
        selected_features = st.multiselect(
            "Select 2 features for decision regions",
            available_features,
            default=available_features[:2]
        )

        if len(selected_features) != 2:
            st.warning("Please select exactly two features.")
        else:
            from sklearn.preprocessing import LabelEncoder
            from mlxtend.plotting import plot_decision_regions

            # Training data
            X_train_sel = results["X_train"][selected_features].to_numpy()
            y_train_sel = results["y_train"].to_numpy()
            le = LabelEncoder()
            y_train_encoded = le.fit_transform(y_train_sel)

            model_for_plot = DecisionTreeClassifier(**results["model"].get_params())
            model_for_plot.fit(X_train_sel, y_train_encoded)

            fig_train, ax_train = plt.subplots(figsize=(6, 4))
            plot_decision_regions(X_train_sel, y_train_encoded, clf=model_for_plot, legend=2, ax=ax_train)
            ax_train.set_xlabel(selected_features[0])
            ax_train.set_ylabel(selected_features[1])
            ax_train.set_title("Decision Regions - Training Data")
            st.pyplot(fig_train)
            download_plot(fig_train, "decision_regions_train")

            # Test data
            X_test_sel = results["X_test"][selected_features].to_numpy()
            y_test_sel = results["y_test"].to_numpy()
            y_test_encoded = le.transform(y_test_sel)

            fig_test, ax_test = plt.subplots(figsize=(6, 4))
            plot_decision_regions(X_test_sel, y_test_encoded, clf=model_for_plot, legend=2, ax=ax_test)
            ax_test.set_xlabel(selected_features[0])
            ax_test.set_ylabel(selected_features[1])
            ax_test.set_title("Decision Regions - Test Data")
            st.pyplot(fig_test)
            download_plot(fig_test, "decision_regions_test")

    if "Learning Curve" in analysis_options:
        # st.write("📈 **Learning Curve**")

        # Get model and data
        X = results["X_train"].to_numpy()
        y = results["y_train"].to_numpy()

        # Convert y to integer if not already
        if y.dtype.kind not in {'i', 'u'}:
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y = le.fit_transform(y)
        else:
            y = y.astype(int)

        # Calculate learning curve
        from sklearn.model_selection import learning_curve
        train_sizes, train_scores, val_scores = learning_curve(
            estimator=results["model"],  # DecisionTreeClassifier is already in results
            X=X,
            y=y,
            cv=5,
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring="accuracy",
            n_jobs=-1
        )

        # Calculate mean and std
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)

        # Plot
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(train_sizes, train_mean, 'o-', label="Training score")

        ax.plot(train_sizes, val_mean, 'o-', label="Validation score")

        ax.set_xlabel("Training set size")
        ax.set_ylabel("Accuracy")
        ax.set_title("Learning Curve (Decision Tree)")
        ax.legend(loc="best")
        ax.grid(True)
        st.pyplot(fig)
        download_plot(fig, "learning_curve")


def decision_tree_page(df) -> None:
    st.title("Decision Tree Classifier")
    train_decision_tree(df)
    st.markdown("---")
    decision_tree_analysis()


def train_knn(df) -> None:
    st.subheader("🔹 Model Settings")
    
    # Select target column
    target = st.selectbox(
        "Select Target Column", 
        options=df.columns, 
        index=list(df.columns).index('label') if 'label' in df.columns else 0
    )
    
    # Select features
    features = st.multiselect(
        "Select Features for the Model", 
        options=[col for col in df.columns if col != target],
        default=[col for col in df.columns if col != target]
    )
    
    # Select encoding type
    encoding_type = st.radio("Encoding Type", ["One-Hot Encoding", "Label Encoding"])
    # Scaler option
    scaler_option = st.selectbox(
        "Scaling Method",
        ["StandardScaler (Z-Score)", "MinMaxScaler", "MaxAbsScaler", "None"]
    )
    
    # Parameter selection
    param_mode = st.radio(
        "Parameter Selection",
        ["Manual", "Automatic (with GridSearchCV)"]
    )
    
    # KNN parameter
    if param_mode == "Manual":
        k = st.slider("Select K Value (Number of Neighbors)", 1, 20, 5)
        metric = st.selectbox("Distance Metric", ["euclidean", "minkowski", "manhattan", "chebyshev"])
    else:
        k = None
        metric = None
    
    # Train-test settings
    test_size = st.slider("Test Set Ratio", 0.1, 0.5, 0.2, 0.05)
    random_seed = st.number_input("Random Seed", value=42)
    shuffle_data = st.checkbox("Shuffle Data", value=True)
    use_stratify = st.checkbox("Stratify (Preserve Class Distribution)", value=True)

    if st.button("Train Model"):
        if not features:
            st.warning("You must select at least one feature.")
            return
        
        # Encoding
        df_encoded = encode_features(df, encoding_type, target_col=target)
        # Update feature names after encoding
        encoded_feature_options = [col for col in df_encoded.columns if col not in target]
        features = encoded_feature_options
        X = df_encoded[features]
        y = df[target]

        # Stratify setting
        stratify_param = y if use_stratify else None
        
        # Split into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_seed, shuffle=shuffle_data, stratify=stratify_param
        )
        
        # Save original data
        X_train_original = X_train.copy()
        X_test_original = X_test.copy()
        
        # Scaler selection
        scaler = None
        if scaler_option == "StandardScaler (Z-Score)":
            scaler = StandardScaler()
        elif scaler_option == "MinMaxScaler":
            scaler = MinMaxScaler()
        elif scaler_option == "MaxAbsScaler":
            scaler = MaxAbsScaler()
        
        # Apply scaling
        if scaler is not None:
            X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=features, index=X_train.index)
            X_test = pd.DataFrame(scaler.transform(X_test), columns=features, index=X_test.index)
        
        if param_mode == "Automatic (with GridSearchCV)":
            from sklearn.model_selection import GridSearchCV

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

            # Transfer results to DataFrame
            results = pd.DataFrame(grid.cv_results_)

            best_score = results['mean_test_score'].max()

            # Filter rows with the best score
            best_rows = results[np.isclose(results['mean_test_score'], best_score)]

            # smallest k
            best_k = best_rows['param_n_neighbors'].min()

            # metrics candidates for this k
            metric_candidates = best_rows.loc[best_rows['param_n_neighbors'] == best_k, 'param_metric']

            # select the alphabetically first metric (e.g., euclidean)
            best_metric = sorted(metric_candidates)[0]

            k = best_k
            metric = best_metric

            st.info(f"Automatic best parameters: k={k}, metric={metric}")
        
        # Model
        model = KNeighborsClassifier(n_neighbors=k, metric=metric)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Determine feature types
        feature_types = {}
        unique_values = {}
        for feature in features:
            if df[feature].dtype == "object" or df[feature].nunique() < 10:  # Categorical feature
                feature_types[feature] = "categorical"
                unique_values[feature] = df[feature].unique().tolist()
            else:  # Numerical feature
                feature_types[feature] = "numerical"

        # Save to session
        st.session_state["knn_results"] = {
            "model": model,
            "X_test": X_test,
            "X_train": X_train,
            "X_test_original": X_test_original,
            "X_train_original": X_train_original,
            "y_test": y_test, 
            "y_train": y_train,
            "y_pred": y_pred,
            "features": features,
            "feature_types": feature_types,
            "unique_values": unique_values,
            "scaler": scaler
        }
        
        st.success("✅ Model trained successfully!")
        st.write(f"**Accuracy:** {accuracy_score(y_test, y_pred):.4f}")
        

# -------------------------
# 3. Analysis function
# -------------------------
def knn_analysis() -> None:
    if "knn_results" not in st.session_state:
        st.info("You must train the model first.")
        return
    
    results = st.session_state["knn_results"]
    y_test = results["y_test"]
    y_pred = results["y_pred"]
    
    st.subheader("📊 Analysis Tools")
    analysis_options = st.multiselect(
        "Select analyses you want to visualize",
        ["Classification Report", "Confusion Matrix", "ROC Curve", "Decision Regions", "Learning Curve"],
        default=["Classification Report"]
    )
    
    # --- Classification Report ---
    if "Classification Report" in analysis_options:
        report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        report_df = pd.DataFrame(report_dict).transpose()
        st.write("**Classification Report**")
        st.dataframe(report_df.style.format({"precision": "{:.4f}"}))
    
    # --- Confusion Matrix ---
    if "Confusion Matrix" in analysis_options:
        st.write("📊 **Confusion Matrix**")
        cm_display_type = st.radio("Display Type", ["Raw Count", "Normalized (%)"], horizontal=True)

        cm = confusion_matrix(y_test, y_pred)

        if cm_display_type == "Normalized (%)":
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
            fmt = ".2f"
        else:
            fmt = "d"

        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt=fmt, cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
        download_plot(fig, "confusion_matrix")
    
    # --- ROC Curve ---
    if "ROC Curve" in analysis_options:
        unique_classes = sorted(list(set(y_test)))
        
        if len(unique_classes) == 2:
            pos_label = st.selectbox("Positive class (for ROC)", unique_classes, index=1)
            fpr, tpr, _ = roc_curve(
                y_test, 
                results["model"].predict_proba(results["X_test"])[:, list(results["model"].classes_).index(pos_label)], 
                pos_label=pos_label
            )
            roc_auc = auc(fpr, tpr)
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, color='blue', lw=2, label=f"AUC = {roc_auc:.2f}")
            ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title(f"ROC Curve (Positive: {pos_label})")
            ax.legend(loc="lower right")
            st.pyplot(fig)
            download_plot(fig, "roc_curve")
        
        elif len(unique_classes) > 2:
            selected_classes = st.multiselect(
                "Select two classes you want to compare for ROC",
                unique_classes,
                default=unique_classes[:2]
            )
            if len(selected_classes) == 2:
                mask = y_test.isin(selected_classes)
                y_test_bin = y_test[mask].apply(lambda x: 1 if x == selected_classes[1] else 0)
                y_score = results["model"].predict_proba(results["X_test"][mask])[:, list(results["model"].classes_).index(selected_classes[1])]
                fpr, tpr, _ = roc_curve(y_test_bin, y_score, pos_label=1)
                roc_auc = auc(fpr, tpr)
                fig, ax = plt.subplots()
                ax.plot(fpr, tpr, color='blue', lw=2, label=f"AUC = {roc_auc:.2f}")
                ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")
                ax.set_title(f"ROC Curve (Positive: {selected_classes[1]})")
                ax.legend(loc="lower right")
                st.pyplot(fig)
                download_plot(fig, "roc_curve")
            else:
                st.warning("Please select exactly two classes.")
        else:
            st.warning("Not enough classes available for ROC Curve.")

    # --- Decision Regions ---
    if "Decision Regions" in analysis_options:
        import copy
        st.subheader("🗺 **Decision Regions** (Can only be plotted with 2 features)")
        
        # Get feature names
        feature_names = results["X_train"].columns.tolist()
        
        # Let user make a selection
        selected_features = st.multiselect(
            "Select two features", 
            feature_names,
            default=feature_names[:2]
        )
        
        if len(selected_features) != 2:
            st.warning("You must select exactly two features.")
        else:
            # --- TRAIN ---
            X_train_sel = results["X_train"][selected_features].to_numpy()
            y_train_sel = results["y_train"].to_numpy()
            
            # Convert y to integer if not already
            if y_train_sel.dtype.kind not in {'i', 'u'}:
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                y_train_sel = le.fit_transform(y_train_sel)
            else:
                y_train_sel = y_train_sel.astype(int)
            
            # Make a deep copy of the original model and retrain
            decision_region_model = copy.deepcopy(results["model"])
            decision_region_model.fit(X_train_sel, y_train_sel)
            
            # Training plot
            fig_train, ax_train = plt.subplots(figsize=(6, 4))
            plot_decision_regions(X_train_sel, y_train_sel, clf=decision_region_model, legend=2, ax=ax_train)
            ax_train.set_xlabel(selected_features[0])
            ax_train.set_ylabel(selected_features[1])
            ax_train.set_title("Decision Regions - Training Data")
            st.pyplot(fig_train)
            download_plot(fig_train, "decision_regions_train")

            # --- TEST ---
            X_test_sel = results["X_test"][selected_features].to_numpy()
            y_test_sel = results["y_test"].to_numpy()
            
            # Convert y to integer if not already
            if y_test_sel.dtype.kind not in {'i', 'u'}:
                if 'le' in locals():  # Use the encoder created in training if available
                    y_test_sel = le.transform(y_test_sel)
                else:
                    le = LabelEncoder()
                    y_test_sel = le.fit_transform(y_test_sel)
            else:
                y_test_sel = y_test_sel.astype(int)
            
            # Test plot
            fig_test, ax_test = plt.subplots(figsize=(6, 4))
            plot_decision_regions(X_test_sel, y_test_sel, clf=decision_region_model, legend=2, ax=ax_test)
            ax_test.set_xlabel(selected_features[0])
            ax_test.set_ylabel(selected_features[1])
            ax_test.set_title("Decision Regions - Test Data")
            st.pyplot(fig_test)
            download_plot(fig_test, "decision_regions_test")
    

    # --- Learning Curve ---
    if "Learning Curve" in analysis_options:
        # st.write("📈 **Learning Curve**")
        
        # Get model and data
        X = results["X_train"].to_numpy()
        y = results["y_train"].to_numpy()

        # Convert y to integer if not already
        if y.dtype.kind not in {'i', 'u'}:
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y = le.fit_transform(y)
        else:
            y = y.astype(int)

        # Calculate learning curve
        # Note: learning_curve actually returns CV validation scores.
        train_sizes, train_scores, val_scores = learning_curve(
            estimator=results["model"],
            X=X,
            y=y,
            cv=5,
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring="accuracy",
            n_jobs=-1
        )

        # Calculate mean and std
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)

        # Plot
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(train_sizes, train_mean, 'o-', label="Training score")
        # ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)

        ax.plot(train_sizes, val_mean, 'o-', label="Validation score")
        # ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1)

        ax.set_xlabel("Training set size")
        ax.set_ylabel("Accuracy")
        ax.set_title("Learning Curve")
        ax.legend(loc="best")
        ax.grid(True)
        st.pyplot(fig)
        download_plot(fig, "learning_curve")
    

def knn_page(df) -> None:
    st.title("KNN Classifier")
    train_knn(df)
    st.markdown("---")
    knn_analysis()


def train_random_forest(df) -> None:
    st.subheader("🔹 Model Settings")
    target = st.selectbox(
        "Select Target Column", 
        options=df.columns, 
        index=list(df.columns).index('label') if 'label' in df.columns else 0
    )
    features = st.multiselect(
        "Select Features for the Model", 
        options=[col for col in df.columns if col != target],
        default=[col for col in df.columns if col != target]
    )
    encoding_type = st.radio("Encoding Type", ["One-Hot Encoding", "Label Encoding"])
    test_size = st.slider("Test Set Ratio", 0.1, 0.5, 0.2, 0.05)
    random_seed = st.number_input("Random Seed", value=42)
    shuffle_data = st.checkbox("Shuffle Data", value=True)
    use_stratify = st.checkbox("Stratify (Preserve Class Distribution)", value=True)
    n_estimators = st.slider("Number of Trees (n_estimators)", 10, 200, 100, 10)
    max_depth = st.slider("Maximum Depth", 1, 20, 5)
    criterion = st.selectbox("Splitting Criterion", ["gini", "entropy", "log_loss"])
    if st.button("Train Model"):
        if not features:
            st.warning("You must select at least one feature.")
            return
        df_encoded = encode_features(df, encoding_type, target_col=target)
        X = df_encoded[features]
        y = df[target]
        stratify_param = y if use_stratify else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_seed, shuffle=shuffle_data, stratify=stratify_param
        )
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            criterion=criterion,
            random_state=random_seed
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Determine feature types
        feature_types = {}
        unique_values = {}
        for feature in features:
            if df[feature].dtype == "object" or df[feature].nunique() < 10:  # Categorical feature
                feature_types[feature] = "categorical"
                unique_values[feature] = df[feature].unique().tolist()
            else:  # Numerical feature
                feature_types[feature] = "numerical"

        # Save to session
        st.session_state["rf_results"] = {
            "model": model,
            "X_test": X_test,
            "X_train": X_train,
            "y_test": y_test, 
            "y_train": y_train,
            "y_pred": y_pred,
            "features": features,
            "feature_types": feature_types,
            "unique_values": unique_values
        }

def random_forest_analysis() -> None:
    if "rf_results" not in st.session_state:
        st.warning("⚠ You must train the model first.")
        return
    results = st.session_state["rf_results"]
    y_test = results["y_test"]
    y_pred = results["y_pred"]
    st.success("✅ Random Forest model trained successfully!")
    st.write(f"**Accuracy:** {accuracy_score(y_test, y_pred):.4f}")
    st.subheader("📊 Analysis Tools")
    analysis_options = st.multiselect(
        "Select analyses you want to visualize",
        ["Classification Report", "Confusion Matrix", "ROC Curve", "Feature Importance", "Decision Regions", "Learning Curve"],
        default=["Classification Report"]
    )
    if "Classification Report" in analysis_options:
        report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True, zero_division=0)).transpose()
        st.dataframe(report_df)
    if "Confusion Matrix" in analysis_options:
        st.write("📊 **Confusion Matrix**")
        cm_display_type = st.radio("Display Type", ["Raw Count", "Normalized (%)"], horizontal=True)
        cm = confusion_matrix(y_test, y_pred)
        if cm_display_type == "Normalized (%)":
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
            fmt = ".2f"
        else:
            fmt = "d"
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt=fmt, cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
        download_plot(fig, "confusion_matrix")
    if "ROC Curve" in analysis_options:
        unique_classes = sorted(list(set(y_test)))
        if len(unique_classes) == 2:
            pos_label = st.selectbox("Positive class (for ROC)", unique_classes, index=1)
            fpr, tpr, _ = roc_curve(
                y_test, 
                results["model"].predict_proba(results["X_test"])[:, list(results["model"].classes_).index(pos_label)], 
                pos_label=pos_label
            )
            roc_auc = auc(fpr, tpr)
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, color='blue', lw=2, label=f"AUC = {roc_auc:.2f}")
            ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title(f"ROC Curve (Positive: {pos_label})")
            ax.legend(loc="lower right")
            st.pyplot(fig)
            download_plot(fig, "roc_curve")
        elif len(unique_classes) > 2:
            selected_classes = st.multiselect(
                "Select two classes you want to compare for ROC",
                unique_classes,
                default=unique_classes[:2]
            )
            if len(selected_classes) == 2:
                mask = y_test.isin(selected_classes)
                y_test_bin = y_test[mask].apply(lambda x: 1 if x == selected_classes[1] else 0)
                y_score = results["model"].predict_proba(results["X_test"][mask])[:, list(results["model"].classes_).index(selected_classes[1])]
                fpr, tpr, _ = roc_curve(y_test_bin, y_score, pos_label=1)
                roc_auc = auc(fpr, tpr)
                fig, ax = plt.subplots()
                ax.plot(fpr, tpr, color='blue', lw=2, label=f"AUC = {roc_auc:.2f}")
                ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")
                ax.set_title(f"ROC Curve (Positive: {selected_classes[1]})")
                ax.legend(loc="lower right")
                st.pyplot(fig)
                download_plot(fig, "roc_curve")
            else:
                st.warning("Please select exactly two classes.")
        else:
            st.warning("Not enough classes available for ROC Curve.")
    if "Feature Importance" in analysis_options:
        importances = results["model"].feature_importances_
        feature_names = results["features"]
        fi_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
        fi_df = fi_df.sort_values("Importance", ascending=False)
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(x="Importance", y="Feature", data=fi_df, ax=ax, hue="Feature", legend=False)
        ax.set_title("Feature Importances (Random Forest)")
        st.pyplot(fig)
        download_plot(fig, "feature_importance")
        st.dataframe(fi_df)
    if "Decision Regions" in analysis_options:
        available_features = list(results["X_test"].columns)
        selected_features = st.multiselect(
            "Select 2 features for decision regions",
            available_features,
            default=available_features[:2]
        )
        if len(selected_features) != 2:
            st.warning("Please select exactly two features.")
        else:
            from sklearn.preprocessing import LabelEncoder
            from mlxtend.plotting import plot_decision_regions
            X_train_sel = results["X_train"][selected_features].to_numpy()
            y_train_sel = results["y_train"].to_numpy()
            le = LabelEncoder()
            y_train_encoded = le.fit_transform(y_train_sel)
            model_for_plot = RandomForestClassifier(**results["model"].get_params())
            model_for_plot.fit(X_train_sel, y_train_encoded)
            fig_train, ax_train = plt.subplots(figsize=(6, 4))
            plot_decision_regions(X_train_sel, y_train_encoded, clf=model_for_plot, legend=2, ax=ax_train)
            ax_train.set_xlabel(selected_features[0])
            ax_train.set_ylabel(selected_features[1])
            ax_train.set_title("Decision Regions - Training Data")
            st.pyplot(fig_train)
            download_plot(fig_train, "decision_regions_train")
            X_test_sel = results["X_test"][selected_features].to_numpy()
            y_test_sel = results["y_test"].to_numpy()
            y_test_encoded = le.transform(y_test_sel)
            fig_test, ax_test = plt.subplots(figsize=(6, 4))
            plot_decision_regions(X_test_sel, y_test_encoded, clf=model_for_plot, legend=2, ax=ax_test)
            ax_test.set_xlabel(selected_features[0])
            ax_test.set_ylabel(selected_features[1])
            ax_test.set_title("Decision Regions - Test Data")
            st.pyplot(fig_test)
            download_plot(fig_test, "decision_regions_test")
    if "Learning Curve" in analysis_options:
        X = results["X_train"].to_numpy()
        y = results["y_train"].to_numpy()
        if y.dtype.kind not in {'i', 'u'}:
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y = le.fit_transform(y)
        else:
            y = y.astype(int)
        from sklearn.model_selection import learning_curve
        train_sizes, train_scores, val_scores = learning_curve(
            estimator=results["model"],
            X=X,
            y=y,
            cv=5,
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring="accuracy",
            n_jobs=-1
        )
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(train_sizes, train_mean, 'o-', label="Training score")
        ax.plot(train_sizes, val_mean, 'o-', label="Validation score")
        ax.set_xlabel("Training set size")
        ax.set_ylabel("Accuracy")
        ax.set_title("Learning Curve (Random Forest)")
        ax.legend(loc="best")
        ax.grid(True)
        st.pyplot(fig)
        download_plot(fig, "learning_curve")
    

def random_forest_page(df) -> None:
    st.title("Random Forest Classifier")
    train_random_forest(df)
    st.markdown("---")
    random_forest_analysis()


def train_xgboost_classifier(df) -> None:
    st.subheader("🔹 Model Settings (XGBoost)")
    target = st.selectbox(
        "Select Target Column", 
        options=df.columns, 
        index=list(df.columns).index('label') if 'label' in df.columns else 0
    )
    features = st.multiselect(
        "Select Features for the Model", 
        options=[col for col in df.columns if col != target],
        default=[col for col in df.columns if col != target]
    )
    encoding_type = st.radio("Encoding Type", ["One-Hot Encoding", "Label Encoding"])
    test_size = st.slider("Test Set Ratio", 0.1, 0.5, 0.2, 0.05)
    random_seed = st.number_input("Random Seed", value=42)
    shuffle_data = st.checkbox("Shuffle Data", value=True)
    use_stratify = st.checkbox("Stratify (Preserve Class Distribution)", value=True)
    n_estimators = st.slider("Number of Trees (n_estimators)", 10, 200, 100, 10)
    max_depth = st.slider("Maximum Depth", 1, 20, 5)
    learning_rate = st.slider("Learning Rate", 0.01, 0.5, 0.1, 0.01)
    if st.button("Train Model"):
        if not features:
            st.warning("You must select at least one feature.")
            return
        df_encoded = encode_features(df, encoding_type, target_col=target)
        X = df_encoded[features]
        y = df[target]
        # Label encoding for y
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        stratify_param = y_encoded if use_stratify else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=random_seed, shuffle=shuffle_data, stratify=stratify_param
        )
        model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_seed,
            eval_metric='mlogloss'
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Determine feature types
        feature_types = {}
        unique_values = {}
        for feature in features:
            if df[feature].dtype == "object" or df[feature].nunique() < 10:  # Categorical feature
                feature_types[feature] = "categorical"
                unique_values[feature] = df[feature].unique().tolist()
            else:  # Numerical feature
                feature_types[feature] = "numerical"

        # Save to session
        st.session_state["xgb_results"] = {
            "model": model,
            "X_test": X_test,
            "X_train": X_train,
            "y_test": y_test,
            "y_train": y_train,
            "y_pred": y_pred,
            "features": features,
            "label_encoder": le,
            "feature_types": feature_types,
            "unique_values": unique_values
        }

def xgboost_classifier_analysis() -> None:
    if "xgb_results" not in st.session_state:
        st.warning("⚠ You must train the model first.")
        return
    results = st.session_state["xgb_results"]
    le = results["label_encoder"]
    y_test = pd.Series(le.inverse_transform(results["y_test"]))
    y_pred = pd.Series(le.inverse_transform(results["y_pred"]))
    st.success("✅ XGBoost model trained successfully!")
    st.write(f"**Accuracy:** {accuracy_score(y_test, y_pred):.4f}")
    st.subheader("📊 Analysis Tools")
    analysis_options = st.multiselect(
        "Select analyses you want to visualize",
        ["Classification Report", "Confusion Matrix", "ROC Curve", "Feature Importance", "Tree Plot", "Decision Regions", "Learning Curve"],
        default=["Classification Report"]
    )
    if "Classification Report" in analysis_options:
        report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True, zero_division=0)).transpose()
        st.dataframe(report_df)
    if "Confusion Matrix" in analysis_options:
        st.write("📊 **Confusion Matrix**")
        cm_display_type = st.radio("Display Type", ["Raw Count", "Normalized (%)"], horizontal=True)
        cm = confusion_matrix(y_test, y_pred)
        if cm_display_type == "Normalized (%)":
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
            fmt = ".2f"
        else:
            fmt = "d"
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt=fmt, cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
        download_plot(fig, "confusion_matrix")
    if "ROC Curve" in analysis_options:
        unique_classes = sorted(list(set(y_test)))
        if len(unique_classes) == 2:
            pos_label = st.selectbox("Positive class (for ROC)", unique_classes, index=1)
            fpr, tpr, _ = roc_curve(
                y_test, 
                results["model"].predict_proba(results["X_test"])[:, list(results["model"].classes_).index(pos_label)], 
                pos_label=pos_label
            )
            roc_auc = auc(fpr, tpr)
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, color='blue', lw=2, label=f"AUC = {roc_auc:.2f}")
            ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title(f"ROC Curve (Positive: {pos_label})")
            ax.legend(loc="lower right")
            st.pyplot(fig)
            download_plot(fig, "roc_curve")
        elif len(unique_classes) > 2:
            selected_classes = st.multiselect(
                "Select two classes you want to compare for ROC",
                unique_classes,
                default=unique_classes[:2]
            )
            if len(selected_classes) == 2:
                # Align indexes
                y_test = y_test.reset_index(drop=True)
                X_test = results["X_test"].reset_index(drop=True)
                mask = y_test.isin(selected_classes).to_numpy()
                # Binary encoding for XGBoost ROC
                y_test_bin = y_test[mask].apply(lambda x: 1 if x == selected_classes[1] else 0)
                class_int = le.transform([selected_classes[1]])[0]
                y_score = results["model"].predict_proba(X_test[mask])[:, list(results["model"].classes_).index(class_int)]
                fpr, tpr, _ = roc_curve(y_test_bin, y_score, pos_label=1)
                roc_auc = auc(fpr, tpr)
                fig, ax = plt.subplots()
                ax.plot(fpr, tpr, color='blue', lw=2, label=f"AUC = {roc_auc:.2f}")
                ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")
                ax.set_title(f"ROC Curve (Positive: {selected_classes[1]})")
                ax.legend(loc="lower right")
                st.pyplot(fig)
                download_plot(fig, "roc_curve")
            else:
                st.warning("Please select exactly two classes.")
        else:
            st.warning("Not enough classes available for ROC Curve.")

    if "Feature Importance" in analysis_options:
        st.write("XGBoost Feature Importances (Gain, Weight, Cover)")
        booster = results["model"].get_booster()
        for importance_type in ["weight", "gain", "cover"]:
            importances = booster.get_score(importance_type=importance_type)
            fi_df = pd.DataFrame(list(importances.items()), columns=["Feature", "Importance"])
            fi_df = fi_df.sort_values("Importance", ascending=False)
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.barplot(x="Importance", y="Feature", data=fi_df, ax=ax, hue="Feature", legend=False)
            ax.set_title(f"Feature Importances ({importance_type})")
            st.pyplot(fig)
            download_plot(fig, f"feature_importance_{importance_type}")
            st.dataframe(fi_df)
    # if "Tree Plot" in analysis_options:
    #     st.write("XGBoost Tree Plot (first tree)")
    #     fig, ax = plt.subplots(figsize=(12, 6))
    #     xgb.plot_tree(results["model"], num_trees=0, ax=ax)
    #     st.pyplot(fig)
    #     download_plot(fig, "xgboost_tree")
    if "Decision Regions" in analysis_options:
        available_features = list(results["X_test"].columns)
        selected_features = st.multiselect(
            "Select 2 features for decision regions",
            available_features,
            default=available_features[:2]
        )
        if len(selected_features) != 2:
            st.warning("Please select exactly two features.")
        else:
            from sklearn.preprocessing import LabelEncoder
            from mlxtend.plotting import plot_decision_regions
            X_train_sel = results["X_train"][selected_features].to_numpy()
            y_train_sel = results["y_train"]
            le = LabelEncoder()
            y_train_encoded = le.fit_transform(y_train_sel)
            model_for_plot = xgb.XGBClassifier(**results["model"].get_params())
            model_for_plot.fit(X_train_sel, y_train_encoded)
            fig_train, ax_train = plt.subplots(figsize=(6, 4))
            plot_decision_regions(X_train_sel, y_train_encoded, clf=model_for_plot, legend=2, ax=ax_train)
            ax_train.set_xlabel(selected_features[0])
            ax_train.set_ylabel(selected_features[1])
            ax_train.set_title("Decision Regions - Training Data")
            st.pyplot(fig_train)
            download_plot(fig_train, "decision_regions_train")
            X_test_sel = results["X_test"][selected_features].to_numpy()
            y_test_sel = results["y_test"]
            y_test_encoded = le.transform(y_test_sel)
            fig_test, ax_test = plt.subplots(figsize=(6, 4))
            plot_decision_regions(X_test_sel, y_test_encoded, clf=model_for_plot, legend=2, ax=ax_test)
            ax_test.set_xlabel(selected_features[0])
            ax_test.set_ylabel(selected_features[1])
            ax_test.set_title("Decision Regions - Test Data")
            st.pyplot(fig_test)
            download_plot(fig_test, "decision_regions_test")
    if "Learning Curve" in analysis_options:
        X = results["X_train"].to_numpy()
        y = results["y_train"]
        if y.dtype.kind not in {'i', 'u'}:
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y = le.fit_transform(y)
        else:
            y = y.astype(int)
        from sklearn.model_selection import learning_curve
        train_sizes, train_scores, val_scores = learning_curve(
            estimator=results["model"],
            X=X,
            y=y,
            cv=5,
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring="accuracy",
            n_jobs=-1
        )
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(train_sizes, train_mean, 'o-', label="Training score")
        ax.plot(train_sizes, val_mean, 'o-', label="Validation score")
        ax.set_xlabel("Training set size")
        ax.set_ylabel("Accuracy")
        ax.set_title("Learning Curve (XGBoost)")
        ax.legend(loc="best")
        ax.grid(True)
        st.pyplot(fig)
        download_plot(fig, "learning_curve")
    

def xgboost_classifier_page(df) -> None:
    st.title("XGBoost Classifier")
    train_xgboost_classifier(df)
    st.markdown("---")
    xgboost_classifier_analysis()
