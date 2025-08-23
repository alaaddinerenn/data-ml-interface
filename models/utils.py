from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, LabelEncoder
import pandas as pd
import streamlit as st

def predict_with_model(model_type: str, results: dict) -> None:
    """
    Allows the model to make predictions based on user input and displays scaled values.

    Args:
        model_type (str): Model type ("classification" or "regression").
        results (dict): Training results of the model (retrieved from session_state).
    """
    st.subheader("🔮 Make a Prediction with a New Example")

    # Get input from the user
    input_data = {}
    for feature in results["features"]:
        if "feature_types" in results and results["feature_types"].get(feature) == "categorical":
            # Selectbox for categorical features
            unique_values = results["unique_values"][feature]
            value = st.selectbox(f"Select value for {feature}:", unique_values, key=f"input_{feature}")
        else:
            # Number input for numerical features
            value = st.number_input(f"Enter value for {feature}:", key=f"input_{feature}")
        input_data[feature] = value

    if st.button("Make Prediction"):
        # Convert input to DataFrame
        input_df = pd.DataFrame([input_data])

        # Check for scaler
        scaler = results.get("scaler", None)
        if scaler is not None:
            st.write(f"Mean: {scaler.mean_ if hasattr(scaler, 'mean_') else 'None'}")
            st.write(f"Scale: {scaler.scale_ if hasattr(scaler, 'scale_') else 'None'}")
            scaled_input_df = pd.DataFrame(scaler.transform(input_df), columns=input_df.columns)
            st.write("🔍 Scaled Values:")
            st.dataframe(scaled_input_df)
        else:
            scaled_input_df = input_df

        # Make prediction
        model = results["model"]
        if model_type == "classification":
            prediction = model.predict(scaled_input_df)[0]
            probabilities = model.predict_proba(scaled_input_df)[0]
            st.write(f"🔹 Predicted Class: **{prediction}**")
            st.write("🔹 Class Probabilities:")
            for i, prob in enumerate(probabilities):
                st.write(f"  - Class {model.classes_[i]}: {prob:.4f}")
        elif model_type == "regression":
            prediction = model.predict(scaled_input_df)[0]
            if isinstance(prediction, (int, float)):
                st.write(f"🔹 Predicted Value: **{prediction:.4f}**")
            else:
                st.write(f"🔹 Predicted Value: **{prediction}**")
        else:
            st.error("Invalid model type. Must be 'classification' or 'regression'.")


# -------------------------
# 1. Encoding function
# -------------------------
def encode_features(df, encoding_type="One-Hot Encoding", target_col=None):
    df_encoded = df.copy()
    
    if encoding_type == "One-Hot Encoding":
        # Keep target_col separate
        target = None
        if target_col:
            if isinstance(target_col, list):
                target = df_encoded[target_col]
                df_encoded = df_encoded.drop(columns=target_col)
            elif target_col in df_encoded.columns:
                target = df_encoded[target_col]
                df_encoded = df_encoded.drop(columns=[target_col])
        df_encoded = pd.get_dummies(df_encoded, drop_first=False)
        # Add the target back
        if target is not None:
            df_encoded = pd.concat([df_encoded, target], axis=1)

    elif encoding_type == "Label Encoding":
        le = LabelEncoder()
        for col in df_encoded.select_dtypes(include=['object', 'category']).columns:
            if col != target_col:
                df_encoded[col] = le.fit_transform(df_encoded[col])

    return df_encoded


# -------------------------
# Scaler selection
# -------------------------
def get_scaler(scaler_name):
    if scaler_name == "Standard Scaler (Z-Score)":
        return StandardScaler()
    elif scaler_name == "Min-Max Scaler":
        return MinMaxScaler()
    elif scaler_name == "Max-Abs Scaler":
        return MaxAbsScaler()
    # robust scaling
    # Additional scaler descriptions can be added
    return None