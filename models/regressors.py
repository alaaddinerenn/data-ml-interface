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
from sklearn.multioutput import MultiOutputRegressor
from utils import download_plot
from models.utils import encode_features, predict_with_model



def train_linear_regressor(df) -> None:
    st.subheader("🔹 Model Settings")

    target = st.selectbox(
        "Select Target Variable", 
        options=df.columns, 
        index=list(df.columns).index('target') if 'target' in df.columns else 0
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

    if st.button("Train Model"):
        if not features:
            st.warning("You must select at least one feature.")
            return

        # Apply encoding
        df_encoded = encode_features(df, encoding_type, target_col=target)
        # Update feature names after encoding
        encoded_feature_options = [col for col in df_encoded.columns if col not in target]
        features = st.multiselect(
            "Select Features for the Model",
            options=encoded_feature_options,
            default=encoded_feature_options
        )
        X = df_encoded[features]
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_seed, shuffle=shuffle_data
        )

        model = LinearRegression()
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

        # Add results to session
        st.session_state["linreg_results"] = {
            "model": model,
            "X_test": X_test,
            "y_test": y_test,
            "y_pred": y_pred,
            "features": features,
            "feature_types": feature_types,
            "unique_values": unique_values
        }

        st.success("✅ Model trained successfully!")
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        st.write(f"**MSE:** {mse:.4f}")
        st.write(f"**R²:** {r2:.4f}")

# -------------------------
# 3. Analysis Function
# -------------------------
def linear_regression_analysis() -> None:
    if "linreg_results" not in st.session_state:
        st.info("You must train the model first.")
        return

    results = st.session_state["linreg_results"]
    y_test = results["y_test"]
    y_pred = results["y_pred"]

    st.subheader("📊 Analysis Tools")
    analysis_options = st.multiselect(
        "Select analyses to visualize",
        ["Actual vs Predicted Scatter", "Error Distribution Histogram", "Prediction Table"],
        default=["Actual vs Predicted Scatter"]
    )

    if "Actual vs Predicted Scatter" in analysis_options:
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        ax.set_xlabel("Actual Values")
        ax.set_ylabel("Predicted Values")
        ax.set_title("Actual vs Predicted")
        st.pyplot(fig)
        download_plot(fig, "prediction_vs_truth")

    if "Error Distribution Histogram" in analysis_options:
        errors = y_test - y_pred
        fig, ax = plt.subplots()
        ax.hist(errors, bins=20, edgecolor='black')
        ax.set_xlabel("Error")
        ax.set_ylabel("Frequency")
        ax.set_title("Error Distribution")
        st.pyplot(fig)
        download_plot(fig, "error_distribution")

    if "Prediction Table" in analysis_options:
        combined = pd.DataFrame({
            'Actual': y_test.values,
            'Predicted': y_pred
        })
        st.write("📄 Prediction and Actual Values Table")
        st.dataframe(combined)

    predict_with_model("regression", results)

# -------------------------
# 4. Page Layout
# -------------------------
def linear_regression_page(df) -> None:
    st.title("Linear Regression")
    train_linear_regressor(df)
    st.markdown("---")
    linear_regression_analysis()



# -------------------------
# 1. Model training function
# -------------------------
def train_sgd_regressor(df) -> None:
    st.subheader("🔹 Model Settings")

    # Target variable (multi-selection)
    target = st.multiselect(
        "Select Target Variable(s)",
        options=df.columns,
        default=[col for col in df.columns if col == "target"]
    )

    # Feature selection
    features = st.multiselect(
        "Select Features for the Model",
        options=[col for col in df.columns if col not in target],
        default=[col for col in df.columns if col not in target]
    )

    # Encoding option
    encoding_type = st.radio("Encoding Type", ["One-Hot Encoding", "Label Encoding"])

    # Scaler option
    scaler_option = st.selectbox(
        "Scaling Method",
        [
            "StandardScaler (Z-Score)",
            "MinMaxScaler",
            "MaxAbsScaler",
            "None"
        ]
    )

    # Test/train settings
    test_size = st.slider("Test Set Ratio", 0.1, 0.5, 0.2, 0.05)
    random_seed = st.number_input("Random Seed", value=42)
    shuffle_data = st.checkbox("Shuffle Data", value=True)

    # SGD parameters
    alpha = st.number_input("Alpha (Regularization Strength)", min_value=0.0001, value=0.0001, step=0.0001, format="%.4f")
    max_iter = st.number_input("Max Iterations", min_value=100, value=1000, step=100)
    early_stopping = st.checkbox("Early Stopping", value=True)
    penalty_choice = st.selectbox("Penalty Type", ["l2", "l1", "elasticnet", "None"])
    penalty = None if penalty_choice == "None" else penalty_choice
    learning_rate = st.selectbox("Learning Rate Schedule", ["constant", "optimal", "invscaling", "adaptive"])
    eta0 = st.number_input("Eta0 (Initial Learning Rate)", min_value=0.00001, value=0.0001, step=0.0001, format="%.4f")

    if st.button("Train Model"):
        if not features:
            st.warning("You must select at least one feature.")
            return
        if not target:
            st.warning("You must select at least one target variable.")
            return

        # Apply encoding
        df_encoded = encode_features(df, encoding_type, target_col=target)
        # Update feature names after encoding
        encoded_feature_options = [col for col in df_encoded.columns if col not in target]
        features = encoded_feature_options
        X = df_encoded[features]
        y = df[target] if len(target) > 1 else df[target[0]]

        # Split into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_seed,
            shuffle=shuffle_data
        )

        # Scaler selection
        scaler = None
        if scaler_option == "StandardScaler (Z-Score)":
            scaler = StandardScaler()
        elif scaler_option == "MinMaxScaler":
            scaler = MinMaxScaler()
        elif scaler_option == "MaxAbsScaler":
            scaler = MaxAbsScaler()
        # scaler = get_scaler(scaler_option)

        # Apply scaling (fit only on train)
        if scaler is not None:
            X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=features, index=X_train.index)
            X_test = pd.DataFrame(scaler.transform(X_test), columns=features, index=X_test.index)

        # Create and train the model
        model = SGDRegressor(
            alpha=alpha,
            max_iter=max_iter,
            penalty=penalty,
            learning_rate=learning_rate,
            eta0=eta0,
            random_state=random_seed,
            early_stopping=early_stopping
        )
        is_multioutput = len(target) > 1
        if is_multioutput:
            model = MultiOutputRegressor(model)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Save results to session_state
        st.session_state["sgd_results"] = {
            "model": model,
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
            "y_pred": y_pred,
            "is_multioutput": is_multioutput,
            "target_names": target
        }

        # Performance metrics
        # st.success("✅ Model trained successfully!")
        # mse = mean_squared_error(y_test, y_pred)
        # r2 = r2_score(y_test, y_pred)
        # st.write(f"**MSE:** {mse:.4f}")
        # st.write(f"**R²:** {r2:.4f}")


# -------------------------
# 2. Analysis function
# -------------------------
def sgd_regression_analysis() -> None:
    if "sgd_results" not in st.session_state:
        st.info("You must train the model first.")
        return

    results = st.session_state["sgd_results"]
    y_test = results["y_test"]
    y_pred = results["y_pred"]
    is_multioutput = results.get("is_multioutput", False)
    target_names = results.get("target_names", None)

    st.success("✅ Model trained successfully!")
    if not is_multioutput:
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        st.write(f"**MSE:** {mse:.4f}")
        st.write(f"**R²:** {r2:.4f}")
    else:
        # Separate metrics for each target in multi-output regression
        for i, col in enumerate(target_names):
            mse = mean_squared_error(y_test[col], y_pred[:, i])
            r2 = r2_score(y_test[col], y_pred[:, i])
            st.write(f"**{col} - MSE:** {mse:.4f}")
            st.write(f"**{col} - R²:** {r2:.4f}")

    st.subheader("📊 Analysis Tools")
    analysis_options = st.multiselect(
        "Select analyses to visualize",
        ["Actual vs Predicted Scatter", "Error Distribution Histogram", "Prediction Table", "Learning Curve"],
        default=["Actual vs Predicted Scatter"]
    )

    if not is_multioutput:
        if "Actual vs Predicted Scatter" in analysis_options:
            fig, ax = plt.subplots()
            ax.scatter(y_test, y_pred)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
            ax.set_xlabel("Actual Values")
            ax.set_ylabel("Predicted Values")
            ax.set_title("Actual vs Predicted")
            st.pyplot(fig)
            download_plot(fig, "prediction_vs_truth")

        if "Error Distribution Histogram" in analysis_options:
            errors = y_test - y_pred
            fig, ax = plt.subplots()
            ax.hist(errors, bins=20, edgecolor='black')
            ax.set_xlabel("Error")
            ax.set_ylabel("Frequency")
            ax.set_title("Error Distribution")
            st.pyplot(fig)
            download_plot(fig, "error_distribution")

        if "Prediction Table" in analysis_options:
            combined = pd.DataFrame({'Actual': y_test.values, 'Predicted': y_pred})
            st.dataframe(combined)
            download_plot(fig, "prediction")
    else:
        # Separate visualizations for each target in multi-output regression
        if "Actual vs Predicted Scatter" in analysis_options:
            for i, col in enumerate(target_names):
                fig, ax = plt.subplots()
                ax.scatter(y_test[col], y_pred[:, i])
                ax.plot([y_test[col].min(), y_test[col].max()], [y_test[col].min(), y_test[col].max()], 'r--')
                ax.set_xlabel("Actual Values")
                ax.set_ylabel("Predicted Values")
                ax.set_title(f"Actual vs Predicted: {col}")
                st.pyplot(fig)
                download_plot(fig, f"prediction_vs_truth_{col}")

        if "Error Distribution Histogram" in analysis_options:
            for i, col in enumerate(target_names):
                errors = y_test[col] - y_pred[:, i]
                fig, ax = plt.subplots()
                ax.hist(errors, bins=20, edgecolor='black')
                ax.set_xlabel("Error")
                ax.set_ylabel("Frequency")
                ax.set_title(f"Error Distribution: {col}")
                st.pyplot(fig)
                download_plot(fig, f"error_distribution_{col}")

        if "Prediction Table" in analysis_options:
            pred_df = pd.DataFrame(y_pred, columns=[f"Predicted_{col}" for col in target_names])
            combined = pd.concat([y_test.reset_index(drop=True), pred_df], axis=1)
            st.dataframe(combined)

    if "Learning Curve" in analysis_options:
        X_train = results["X_train"]
        y_train = results["y_train"]

        train_sizes, train_scores, val_scores = learning_curve(
            estimator=results["model"],
            X=X_train,
            y=y_train,
            cv=5,
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring="r2",
            n_jobs=-1
        )

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', label="Training score")
        ax.plot(train_sizes, np.mean(val_scores, axis=1), 'o-', label="Validation score")
        ax.set_xlabel("Training set size")
        ax.set_ylabel("R² Score")
        ax.set_title("Learning Curve")
        ax.legend(loc="best")
        ax.grid(True)
        st.pyplot(fig)
        download_plot(fig, "learning_curve")
    
    predict_with_model("regression", results)


# -------------------------
# 3. Page Layout
# -------------------------
def sgd_regression_page(df) -> None:
    st.title("SGD Regressor")
    train_sgd_regressor(df)
    st.markdown("---")
    sgd_regression_analysis()