import streamlit as st
from file import load_file

# Yeni OOP yapısından importlar
from models import (
    # Regressors
    linear_regression_page,
    sgd_regression_page,
    
    # Classifiers
    decision_tree_page,
    knn_page,
    random_forest_page,
    xgboost_classifier_page,
    
    # Clusterers
    kmeans_page,
    dbscan_page,
    agglomerative_page
)

st.set_page_config(page_title="Data & ML Interface", page_icon="🤖")

# Title at the top of the page
st.markdown(
    """
    <h1 style="text-align: center; font-size: 50px; margin: 10px 0 30px 0;">
        Data & ML Interface
    </h1>
    """,
    unsafe_allow_html=True
)

st.write("\n\n\n\n\n")
st.title("🤖 Machine Learning Page")

# File upload option
df_uploaded = load_file()

# Determine which data to use
if df_uploaded is not None:
    df = df_uploaded
    # Reset data from analysis page
    st.session_state.df_for_ml_clean = None
    st.session_state.df_for_ml_raw = None
    st.write("📂 Data loaded from file:")
    st.dataframe(df)

elif 'df_for_ml_clean' in st.session_state and st.session_state.df_for_ml_clean is not None:
    df = st.session_state.df_for_ml_clean
    st.write("✅ Cleaned data loaded (from analysis page):")
    st.dataframe(df)

elif 'df_for_ml_raw' in st.session_state and st.session_state.df_for_ml_raw is not None:
    df = st.session_state.df_for_ml_raw
    st.write("✅ Raw data loaded (from analysis page):")
    st.dataframe(df)

else:
    df = None

# If data exists
if df is not None and not df.empty:
    # Select ML task type
    st.markdown(
        """
        <h2 style='text-align: center; font-size: 24px; font-weight: bold;'>
            Select Task Type
        </h2>
        """,
        unsafe_allow_html=True
    )
    
    ml_type = st.selectbox(
        " ",
        options=["Regression", "Classification", "Clustering"],
        index=None,
        help="Select the task type: Regression, Classification, or Clustering",
        label_visibility="collapsed"
    )

    if ml_type:
        st.markdown(
            f"""
            <h2 style='text-align: center; font-size: 24px; font-weight: bold;'>
                Select {ml_type} Algorithm
            </h2>
            """,
            unsafe_allow_html=True
        )

        # ============================================
        # REGRESSION
        # ============================================
        if ml_type == "Regression":
            st.info("📈 Regression: Used for predicting continuous values.")
            
            model_choice = st.selectbox(
                " ",
                options=[
                    "Linear Regression",
                    "Stochastic Gradient Descent Regressor"
                ],
                index=None,
                help="Select a regression algorithm",
                label_visibility="collapsed"
            )
            
            if model_choice == "Linear Regression":
                st.markdown(
                    "<span style='font-size:18px; font-weight:bold;'>"
                    "📊 The simplest linear regression algorithm.</span>",
                    unsafe_allow_html=True
                )
                linear_regression_page(df)
                
            elif model_choice == "Stochastic Gradient Descent Regressor":
                st.markdown(
                    "<span style='font-size:18px; font-weight:bold;'>"
                    "⚡ Fast and efficient regression for large datasets.</span>",
                    unsafe_allow_html=True
                )
                sgd_regression_page(df)

        # ============================================
        # CLASSIFICATION
        # ============================================
        elif ml_type == "Classification":
            st.info("🎯 Classification: Used for categorizing data into classes.")
            
            model_choice = st.selectbox(
                " ",
                options=[
                    "Decision Tree Classifier",
                    "KNN Classifier",
                    "Random Forest Classifier",
                    "XGBoost Classifier"
                ],
                index=None,
                help="Select a classification algorithm",
                label_visibility="collapsed"
            )
            
            if model_choice == "Decision Tree Classifier":
                st.markdown(
                    "<span style='font-size:18px; font-weight:bold;'>"
                    "🌳 Classifies data using decision trees.</span>",
                    unsafe_allow_html=True
                )
                decision_tree_page(df)
                
            elif model_choice == "KNN Classifier":
                st.markdown(
                    "<span style='font-size:18px; font-weight:bold;'>"
                    "👥 Classifies based on the labels of neighboring examples.</span>",
                    unsafe_allow_html=True
                )
                knn_page(df)
                
            elif model_choice == "Random Forest Classifier":
                st.markdown(
                    "<span style='font-size:18px; font-weight:bold;'>"
                    "🌲 Stronger classification by combining multiple decision trees.</span>",
                    unsafe_allow_html=True
                )
                random_forest_page(df)
                
            elif model_choice == "XGBoost Classifier":
                st.markdown(
                    "<span style='font-size:18px; font-weight:bold;'>"
                    "🚀 Powerful and fast classification using gradient boosting.</span>",
                    unsafe_allow_html=True
                )
                xgboost_classifier_page(df)

        # ============================================
        # CLUSTERING
        # ============================================
        elif ml_type == "Clustering":
            st.info("🔍 Clustering: Groups data points based on their similarities.")
            
            model_choice = st.selectbox(
                " ",
                options=[
                    "K-Means",
                    "DBSCAN",
                    "Agglomerative Clustering"
                ],
                index=None,
                help="Select a clustering algorithm",
                label_visibility="collapsed"
            )
            
            if model_choice == "K-Means":
                st.markdown(
                    "<span style='font-size:18px; font-weight:bold;'>"
                    "📍 Divides data points into k clusters.</span>",
                    unsafe_allow_html=True
                )
                kmeans_page(df)
                
            elif model_choice == "DBSCAN":
                st.markdown(
                    "<span style='font-size:18px; font-weight:bold;'>"
                    "🔎 Density-based clustering that can find arbitrary shaped clusters.</span>",
                    unsafe_allow_html=True
                )
                dbscan_page(df)
                
            elif model_choice == "Agglomerative Clustering":
                st.markdown(
                    "<span style='font-size:18px; font-weight:bold;'>"
                    "🌳 Hierarchical clustering using bottom-up approach.</span>",
                    unsafe_allow_html=True
                )
                agglomerative_page(df)

else:
    st.info("📤 Please upload a dataset or send one from the analysis page.")
