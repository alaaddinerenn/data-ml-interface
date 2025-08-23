import streamlit as st
from file import load_file
from models.regressors import linear_regression_page, sgd_regression_page
from models.classifiers import knn_page, decision_tree_page, random_forest_page, xgboost_classifier_page
from models.clusterers import kmeans_page

# Sayfa içeriğinin üstünde başlık (sidebar açık/kapalı fark etmez)
st.markdown(
    """
    <h1 style="position: fixed; top: 40px; left: 20px; margin: 0; z-index: 999;">
        Data & ML Arayüzü
    </h1>
    """,
    unsafe_allow_html=True
)

# Sayfa içeriği biraz aşağıda başlasın diye boşluk bırak
st.write("\n\n\n\n\n")

st.title("🤖 Makine Öğrenmesi Sayfası")


# 📌 Her zaman dosya yükleme seçeneği olsun
df_uploaded = load_file()

# Eğer kullanıcı yeni dosya yüklediyse → onu kullan ve session’daki ML verilerini override et
if df_uploaded is not None:
    df = df_uploaded
    # Analizden gelen verileri sıfırla, çünkü artık yeni dosya var
    st.session_state.df_for_ml_clean = None
    st.session_state.df_for_ml_raw = None
    st.write("📂 Dosyadan veri yüklendi:")
    st.dataframe(df)

# Eğer dosya yüklenmemişse → session’daki ML verilerine bak
elif 'df_for_ml_clean' in st.session_state and st.session_state.df_for_ml_clean is not None:
    df = st.session_state.df_for_ml_clean
    st.write("✅ Temizlenmiş veri yüklendi (analiz sayfasından):")
    st.dataframe(df)

elif 'df_for_ml_raw' in st.session_state and st.session_state.df_for_ml_raw is not None:
    df = st.session_state.df_for_ml_raw
    st.write("✅ Ham veri yüklendi (analiz sayfasından):")
    st.dataframe(df)

else:
    df = None

# Eğer veri varsa
if df is not None and not df.empty:
    # 1️⃣ ML türü seçimi
    st.markdown(
            f"""
            <h2 style='text-align: center; font-size: 24px; font-weight: bold;'>
                Görev Türünü Seçiniz
            </h2>
            """,
            unsafe_allow_html=True
        )
    ml_type = st.selectbox(
        " ",
        options=["Regresyon", "Sınıflandırma", "Kümeleme"],
        index=None,
        help="Görev türünü seçin: Regresyon, Sınıflandırma veya Kümeleme"
    )

    if ml_type:
        st.markdown(
            f"""
            <h2 style='text-align: center; font-size: 24px; font-weight: bold;'>
                {ml_type} Algoritması Seçiniz
            </h2>
            """,
            unsafe_allow_html=True
        )

        # 2️⃣ Algoritma seçimi (ML türüne göre)
        if ml_type == "Regresyon":
            st.info("Regresyon: Sürekli değer tahmini için kullanılır.")
            model_choice = st.selectbox(
                " ",
                options=["Linear Regression", "Stochastic Gradient Descent Regressor"],
                index=None,
                help="Algoritma hakkında bilgi almak için üzerine gelin."
            )
            if model_choice == "Linear Regression":
                st.markdown("<span style='font-size:18px; font-weight:bold;'>En basit doğrusal regresyon algoritmasıdır.</span>", unsafe_allow_html=True)
                linear_regression_page(df)
            elif model_choice == "Stochastic Gradient Descent Regressor":
                st.markdown("<span style='font-size:18px; font-weight:bold;'>Büyük veri setlerinde hızlı ve verimli regresyon sağlar.</span>", unsafe_allow_html=True)
                sgd_regression_page(df)

        elif ml_type == "Sınıflandırma":
            st.info("Sınıflandırma: Verileri kategorilere ayırmak için kullanılır.")
            model_choice = st.selectbox(
                " ",
                options=["KNN Classifier", "Decision Tree Classifier", "Random Forest Classifier", "XGBoost Classifier"],
                index=None,
                label_visibility="collapsed"
            )
            if model_choice == "KNN Classifier":
                st.markdown("<span style='font-size:18px; font-weight:bold;'>Komşu örneklerin etiketlerine göre sınıflandırma yapar.</span>", unsafe_allow_html=True)
                knn_page(df)
            elif model_choice == "Decision Tree Classifier":
                st.markdown("<span style='font-size:18px; font-weight:bold;'>Karar ağaçları ile sınıflandırma yapar.</span>", unsafe_allow_html=True)
                decision_tree_page(df)
            elif model_choice == "Random Forest Classifier":
                st.markdown("<span style='font-size:18px; font-weight:bold;'>Birden fazla karar ağacının birleşimiyle daha güçlü sınıflandırma yapar.</span>", unsafe_allow_html=True)
                random_forest_page(df)
            elif model_choice == "XGBoost Classifier":
                st.markdown("<span style='font-size:18px; font-weight:bold;'>Gradient boosting ile güçlü ve hızlı sınıflandırma sağlar.</span>", unsafe_allow_html=True)
                xgboost_classifier_page(df)

        elif ml_type == "Kümeleme":
            st.info("Kümeleme: Veri noktalarını benzerliklerine göre gruplar.")
            model_choice = st.selectbox(
                " ",
                options=["K-Means"],
                index=None,
                label_visibility="collapsed"
            )
            if model_choice == "K-Means":
                st.markdown("<span style='font-size:18px; font-weight:bold;'>Veri noktalarını k adet kümeye ayırır.</span>", unsafe_allow_html=True)
                kmeans_page(df)
else:
    st.info("📌 Lütfen bir veri seti yükleyin veya analiz sayfasından gönderin.")
