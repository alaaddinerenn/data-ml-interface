from sklearn.preprocessing import LabelEncoder
import pandas as pd


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