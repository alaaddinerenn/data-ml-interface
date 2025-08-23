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
    st.subheader("🔹 Model Ayarları")
    
    target = st.selectbox(
        "Hedef Değişkeni Seçin", 
        options=df.columns, 
        index=list(df.columns).index('target') if 'target' in df.columns else 0
    )
    
    features = st.multiselect(
        "Modelde Kullanılacak Özellikleri Seçin", 
        options=[col for col in df.columns if col != target],
        default=[col for col in df.columns if col != target]
    )
    
    encoding_type = st.radio("Encoding Tipi", ["One-Hot Encoding", "Label Encoding"])
    test_size = st.slider("Test Set Oranı", 0.1, 0.5, 0.2, 0.05)
    random_seed = st.number_input("Random Seed", value=42)
    shuffle_data = st.checkbox("Veriyi Karıştır (Shuffle)", value=True)

    if st.button("Modeli Eğit"):
        if not features:
            st.warning("En az bir özellik seçmelisiniz.")
            return
        
        # Encoding uygula
        df_encoded = encode_features(df, encoding_type, target_col=target)
        # Encoding sonrası feature adlarını güncelle
        encoded_feature_options = [col for col in df_encoded.columns if col not in target]
        features = st.multiselect(
            "Modelde Kullanılacak Özellikleri Seçin",
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

        # Özellik türlerini belirle
        feature_types = {}
        unique_values = {}
        for feature in features:
            if df[feature].dtype == "object" or df[feature].nunique() < 10:  # Kategorik özellik
                feature_types[feature] = "categorical"
                unique_values[feature] = df[feature].unique().tolist()
            else:  # Sayısal özellik
                feature_types[feature] = "numerical"

        # Sonuçlara ekle
        st.session_state["linreg_results"] = {
            "model": model,
            "X_test": X_test,
            "y_test": y_test,
            "y_pred": y_pred,
            "features": features,
            "feature_types": feature_types,
            "unique_values": unique_values
        }

        st.success("✅ Model başarıyla eğitildi!")
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        st.write(f"**MSE:** {mse:.4f}")
        st.write(f"**R²:** {r2:.4f}")

# -------------------------
# 3. Analiz fonksiyonu
# -------------------------
def linear_regression_analysis() -> None:
    if "linreg_results" not in st.session_state:
        st.info("Önce modeli eğitmelisiniz.")
        return

    results = st.session_state["linreg_results"]
    y_test = results["y_test"]
    y_pred = results["y_pred"]

    st.subheader("📊 Analiz Araçları")
    analysis_options = st.multiselect(
        "Görselleştirmek istediğiniz analizleri seçin",
        ["Gerçek vs Tahmin Scatter", "Hata Dağılımı Histogram", "Tahmin Tablosu"],
        default=["Gerçek vs Tahmin Scatter"]
    )

    if "Gerçek vs Tahmin Scatter" in analysis_options:
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        ax.set_xlabel("Gerçek Değerler")
        ax.set_ylabel("Tahmin Edilen Değerler")
        ax.set_title("Gerçek vs Tahmin")
        st.pyplot(fig)
        download_plot(fig, "prediction_vs_truth")

    if "Hata Dağılımı Histogram" in analysis_options:
        errors = y_test - y_pred
        fig, ax = plt.subplots()
        ax.hist(errors, bins=20, edgecolor='black')
        ax.set_xlabel("Hata")
        ax.set_ylabel("Frekans")
        ax.set_title("Hata Dağılımı")
        st.pyplot(fig)
        download_plot(fig, "error_distribution")

    if "Tahmin Tablosu" in analysis_options:
        combined = pd.DataFrame({
            'Actual': y_test.values,
            'Predicted': y_pred
        })
        st.write("📄 Tahmin ve Gerçek Değerler Tablosu")
        st.dataframe(combined)
    
    predict_with_model("regression", results)

# -------------------------
# 4. Sayfa düzeni
# -------------------------
def linear_regression_page(df) -> None:
    st.title("Linear Regression")
    train_linear_regressor(df)
    st.markdown("---")
    linear_regression_analysis()



# -------------------------
# 1. Model eğitim fonksiyonu
# -------------------------
def train_sgd_regressor(df) -> None:
    st.subheader("🔹 Model Ayarları")

    # Hedef değişken (çoklu seçim)
    target = st.multiselect(
        "Hedef Değişken(leri) Seçin",
        options=df.columns,
        default=[col for col in df.columns if col == "target"]
    )

    # Özellik seçimi
    features = st.multiselect(
        "Modelde Kullanılacak Özellikleri Seçin",
        options=[col for col in df.columns if col not in target],
        default=[col for col in df.columns if col not in target]
    )

    # Encoding seçeneği
    encoding_type = st.radio("Encoding Tipi", ["One-Hot Encoding", "Label Encoding"])

    # Scaler seçeneği
    scaler_option = st.selectbox(
        "Ölçeklendirme Yöntemi",
        [
            "StandardScaler (Z-Score)",
            "MinMaxScaler",
            "MaxAbsScaler",
            "Yok"
        ]
    )

    # Test/eğitim ayarları
    test_size = st.slider("Test Set Oranı", 0.1, 0.5, 0.2, 0.05)
    random_seed = st.number_input("Random Seed", value=42)
    shuffle_data = st.checkbox("Veriyi Karıştır (Shuffle)", value=True)

    # SGD parametreleri
    alpha = st.number_input("Alpha (Regularization Strength)", min_value=0.0001, value=0.0001, step=0.0001, format="%.4f")
    max_iter = st.number_input("Max Iterations", min_value=100, value=1000, step=100)
    early_stopping = st.checkbox("Erken Durdurma", value=True)
    penalty_choice = st.selectbox("Penalty Türü", ["l2", "l1", "elasticnet", "None"])
    penalty = None if penalty_choice == "None" else penalty_choice
    learning_rate = st.selectbox("Learning Rate Schedule", ["constant", "optimal", "invscaling", "adaptive"])
    eta0 = st.number_input("Eta0 (Başlangıç Learning Rate)", min_value=0.00001, value=0.0001, step=0.0001, format="%.4f")

    if st.button("Modeli Eğit"):
        if not features:
            st.warning("En az bir özellik seçmelisiniz.")
            return
        if not target:
            st.warning("En az bir hedef değişken seçmelisiniz.")
            return

        # Encoding uygula
        df_encoded = encode_features(df, encoding_type, target_col=target)
        # Encoding sonrası feature adlarını güncelle
        encoded_feature_options = [col for col in df_encoded.columns if col not in target]
        features = encoded_feature_options
        X = df_encoded[features]
        y = df[target] if len(target) > 1 else df[target[0]]

        # Eğitim ve test setine ayır
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_seed,
            shuffle=shuffle_data
        )

        # Scaler seçimi
        scaler = None
        if scaler_option == "StandardScaler (Z-Score)":
            scaler = StandardScaler()
        elif scaler_option == "MinMaxScaler":
            scaler = MinMaxScaler()
        elif scaler_option == "MaxAbsScaler":
            scaler = MaxAbsScaler()
        # scaler = get_scaler(scaler_option)

        # Ölçeklendirme uygula (fit sadece train'e)
        if scaler is not None:
            X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=features, index=X_train.index)
            X_test = pd.DataFrame(scaler.transform(X_test), columns=features, index=X_test.index)

        # Modeli oluştur ve eğit
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

        # Sonuçları session_state'e kaydet
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

        # Performans metrikleri
        # st.success("✅ Model başarıyla eğitildi!")
        # mse = mean_squared_error(y_test, y_pred)
        # r2 = r2_score(y_test, y_pred)
        # st.write(f"**MSE:** {mse:.4f}")
        # st.write(f"**R²:** {r2:.4f}")


# -------------------------
# 2. Analiz fonksiyonu
# -------------------------
def sgd_regression_analysis() -> None:
    if "sgd_results" not in st.session_state:
        st.info("Önce modeli eğitmelisiniz.")
        return

    results = st.session_state["sgd_results"]
    y_test = results["y_test"]
    y_pred = results["y_pred"]
    is_multioutput = results.get("is_multioutput", False)
    target_names = results.get("target_names", None)

    st.success("✅ Model başarıyla eğitildi!")
    if not is_multioutput:
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        st.write(f"**MSE:** {mse:.4f}")
        st.write(f"**R²:** {r2:.4f}")
    else:
        # Çoklu regresyon için her target'a ayrı metrik
        for i, col in enumerate(target_names):
            mse = mean_squared_error(y_test[col], y_pred[:, i])
            r2 = r2_score(y_test[col], y_pred[:, i])
            st.write(f"**{col} - MSE:** {mse:.4f}")
            st.write(f"**{col} - R²:** {r2:.4f}")

    st.subheader("📊 Analiz Araçları")
    analysis_options = st.multiselect(
        "Görselleştirmek istediğiniz analizleri seçin",
        ["Gerçek vs Tahmin Scatter", "Hata Dağılımı Histogram", "Tahmin Tablosu", "Learning Curve"],
        default=["Gerçek vs Tahmin Scatter"]
    )

    if not is_multioutput:
        if "Gerçek vs Tahmin Scatter" in analysis_options:
            fig, ax = plt.subplots()
            ax.scatter(y_test, y_pred)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
            ax.set_xlabel("Gerçek Değerler")
            ax.set_ylabel("Tahmin Edilen Değerler")
            ax.set_title("Gerçek vs Tahmin")
            st.pyplot(fig)
            download_plot(fig, "prediction_vs_truth")

        if "Hata Dağılımı Histogram" in analysis_options:
            errors = y_test - y_pred
            fig, ax = plt.subplots()
            ax.hist(errors, bins=20, edgecolor='black')
            ax.set_xlabel("Hata")
            ax.set_ylabel("Frekans")
            ax.set_title("Hata Dağılımı")
            st.pyplot(fig)
            download_plot(fig, "error_distribution")

        if "Tahmin Tablosu" in analysis_options:
            combined = pd.DataFrame({'Actual': y_test.values, 'Predicted': y_pred})
            st.dataframe(combined)
            download_plot(fig, "prediction")
    else:
        # Çoklu regresyon için her target'a ayrı görselleştirme
        if "Gerçek vs Tahmin Scatter" in analysis_options:
            for i, col in enumerate(target_names):
                fig, ax = plt.subplots()
                ax.scatter(y_test[col], y_pred[:, i])
                ax.plot([y_test[col].min(), y_test[col].max()], [y_test[col].min(), y_test[col].max()], 'r--')
                ax.set_xlabel("Gerçek Değerler")
                ax.set_ylabel("Tahmin Edilen Değerler")
                ax.set_title(f"Gerçek vs Tahmin: {col}")
                st.pyplot(fig)
                download_plot(fig, f"prediction_vs_truth_{col}")

        if "Hata Dağılımı Histogram" in analysis_options:
            for i, col in enumerate(target_names):
                errors = y_test[col] - y_pred[:, i]
                fig, ax = plt.subplots()
                ax.hist(errors, bins=20, edgecolor='black')
                ax.set_xlabel("Hata")
                ax.set_ylabel("Frekans")
                ax.set_title(f"Hata Dağılımı: {col}")
                st.pyplot(fig)
                download_plot(fig, f"error_distribution_{col}")

        if "Tahmin Tablosu" in analysis_options:
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
# 3. Sayfa düzeni
# -------------------------
def sgd_regression_page(df) -> None:
    st.title("SGD Regressor")
    train_sgd_regressor(df)
    st.markdown("---")
    sgd_regression_analysis()