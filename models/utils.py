from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, LabelEncoder
import pandas as pd
import streamlit as st


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