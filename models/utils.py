from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, LabelEncoder
import pandas as pd
import streamlit as st

def predict_with_model(model_type: str, results: dict) -> None:
    """
    Kullanıcıdan input alarak modelin tahmin yapmasını sağlar ve scale edilmiş değerleri gösterir.

    Args:
        model_type (str): Model türü ("classification" veya "regression").
        results (dict): Modelin eğitim sonuçları (session_state'den alınır).
    """
    st.subheader("🔮 Yeni Bir Örnekle Tahmin Yap")

    # Kullanıcıdan input al
    input_data = {}
    for feature in results["features"]:
        if "feature_types" in results and results["feature_types"].get(feature) == "categorical":
            # Kategorik özellikler için selectbox
            unique_values = results["unique_values"][feature]
            value = st.selectbox(f"{feature} değeri seçin:", unique_values, key=f"input_{feature}")
        else:
            # Sayısal özellikler için number_input
            value = st.number_input(f"{feature} değeri girin:", key=f"input_{feature}")
        input_data[feature] = value

    if st.button("Tahmin Yap"):
        # Input'u DataFrame'e çevir
        input_df = pd.DataFrame([input_data])

        # Scaler kontrolü
        scaler = results.get("scaler", None)
        if scaler is not None:
            st.write(f"Mean: {scaler.mean_ if hasattr(scaler, 'mean_') else 'Yok'}")
            st.write(f"Scale: {scaler.scale_ if hasattr(scaler, 'scale_') else 'Yok'}")
            scaled_input_df = pd.DataFrame(scaler.transform(input_df), columns=input_df.columns)
            st.write("🔍 Ölçeklendirilmiş Değerler:")
            st.dataframe(scaled_input_df)
        else:
            scaled_input_df = input_df

        # Tahmin yap
        model = results["model"]
        if model_type == "classification":
            prediction = model.predict(scaled_input_df)[0]
            probabilities = model.predict_proba(scaled_input_df)[0]
            st.write(f"🔹 Tahmin Edilen Sınıf: **{prediction}**")
            st.write("🔹 Sınıf Olasılıkları:")
            for i, prob in enumerate(probabilities):
                st.write(f"  - Sınıf {model.classes_[i]}: {prob:.4f}")
        elif model_type == "regression":
            prediction = model.predict(scaled_input_df)[0]
            if isinstance(prediction, (int, float)):
                st.write(f"🔹 Tahmin Edilen Değer: **{prediction:.4f}**")
            else:
                st.write(f"🔹 Tahmin Edilen Değer: **{prediction}**")
        else:
            st.error("Geçersiz model türü. 'classification' veya 'regression' olmalı.")



# -------------------------
# 1. Encoding fonksiyonu
# -------------------------
def encode_features(df, encoding_type="One-Hot Encoding", target_col=None):
    df_encoded = df.copy()
    
    if encoding_type == "One-Hot Encoding":
        # target_col'u ayrı tut
        target = None
        if target_col:
            if isinstance(target_col, list):
                target = df_encoded[target_col]
                df_encoded = df_encoded.drop(columns=target_col)
            elif target_col in df_encoded.columns:
                target = df_encoded[target_col]
                df_encoded = df_encoded.drop(columns=[target_col])
        df_encoded = pd.get_dummies(df_encoded, drop_first=False)
        # hedefi geri ekle
        if target is not None:
            df_encoded = pd.concat([df_encoded, target], axis=1)

    elif encoding_type == "Label Encoding":
        le = LabelEncoder()
        for col in df_encoded.select_dtypes(include=['object', 'category']).columns:
            if col != target_col:
                df_encoded[col] = le.fit_transform(df_encoded[col])

    return df_encoded


# -------------------------
# Scaler seçimi
# -------------------------
def get_scaler(scaler_name):
    if scaler_name == "Standart Scaler (Z-Score)":
        return StandardScaler()
    elif scaler_name == "Min-Max Scaler":
        return MinMaxScaler()
    elif scaler_name == "Max-Abs Scaler":
        return MaxAbsScaler()
    # robust scaling
    # scaler açıklaması eklenebilir
    return None