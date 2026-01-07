import streamlit as st

st.set_page_config(page_title="Data & ML Interface", page_icon="ℹ️")


# Title at the top of the page (regardless of sidebar visibility)
st.markdown(
    """
    <h1 style="text-align: center; font-size: 50px; margin: 10px 0 30px 0;">
        Data & ML Interface
    </h1>
    """,
    unsafe_allow_html=True
)

# Add space so the page content starts lower
st.write("\n\n\n\n\n")

st.html(
    """
    <div style=\"font-family: Arial, sans-serif; line-height: 1.6; color: #333;\">
        <h1 style=\"text-align: center; font-size: 40px; margin-bottom: 20px;\">ℹ️ About</h1>
        <p style=\"text-align: center; font-size: 22px; color: #555; margin-bottom: 25px;\">
            Welcome to the <strong>Data & ML Interface</strong>! This application empowers users to explore datasets, visualize features, and train machine learning models without writing a single line of code.
        </p>

        <h2 style=\"font-size: 28px; color: #333; margin-bottom: 15px;\">✨ Features</h2>
        <ul style=\"font-size: 20px; color: #444; padding-left: 20px; list-style-type: disc;\">
            <li style=\"margin-bottom: 10px;\">📂 <strong>Data Upload:</strong> Supports CSV, Excel, and TSV file formats.</li>
            <li style=\"margin-bottom: 10px;\">🧹 <strong>Data Cleaning:</strong> Handle missing values and preprocess data easily.</li>
            <li style=\"margin-bottom: 10px;\">📊 <strong>Data Analysis:</strong> Generate statistical summaries and visualize data with histograms, boxplots, and scatter plots.</li>
            <li style=\"margin-bottom: 10px;\">🤖 <strong>Machine Learning:</strong> Train models for classification, regression, and clustering using algorithms like Linear Regression, Decision Trees, and K-Means.</li>
            <li style=\"margin-bottom: 10px;\">📈 <strong>Model Evaluation:</strong> Analyze model performance with metrics like MSE, R², and confusion matrices.</li>
            <li style=\"margin-bottom: 10px;\">📥 <strong>Downloadable Results:</strong> Save generated plots and analysis results.</li>
        </ul>

        <h2 style=\"font-size: 28px; color: #333; margin-bottom: 15px;\">🔧 Technologies Used</h2>
        <ul style=\"font-size: 20px; color: #444; padding-left: 20px; list-style-type: disc;\">
            <li style=\"margin-bottom: 10px;\">Python: Core programming language.</li>
            <li style=\"margin-bottom: 10px;\">Streamlit: Framework for building interactive web applications.</li>
            <li style=\"margin-bottom: 10px;\">Pandas: Data manipulation and analysis.</li>
            <li style=\"margin-bottom: 10px;\">Scikit-learn: Machine learning algorithms and preprocessing.</li>
            <li style=\"margin-bottom: 10px;\">Matplotlib & Seaborn: Data visualization.</li>
        </ul>

        <h2 style=\"font-size: 28px; color: #333; margin-bottom: 15px;\">🚀 Future Enhancements</h2>
        <ul style=\"font-size: 20px; color: #444; padding-left: 20px; list-style-type: disc;\">
            <li style=\"margin-bottom: 10px;\">Implement advanced machine learning algorithms (e.g., Neural Networks, SVMs).</li>
            <li style=\"margin-bottom: 10px;\">Enhance visualization options with interactive plots.</li>
            <li style=\"margin-bottom: 10px;\">Add automated hyperparameter tuning for models.</li>
        </ul>

        <h2 style=\"font-size: 28px; color: #333; margin-bottom: 15px;\">👨‍💻 Developer</h2>
        <p style=\"font-size: 20px; color: #666;\">
            Developed by <strong>Alaaddin Eren Namlı</strong>. Feel free to reach out for feedback or collaboration opportunities!
        </p>
    </div>
    """
)
