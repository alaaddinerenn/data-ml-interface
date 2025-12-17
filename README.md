# Data & ML Interface

A **Streamlit** application designed to empower non-programmers to explore datasets, visualize features, and train machine learning models (classification, regression, clustering) without writing a single line of code.

---

## Features

- **📂 Data Upload**: Supports CSV, Excel, and TSV file formats.
- **🧹 Data Cleaning**: Handle missing values and preprocess data easily.
- **📊 Data Analysis**: Generate statistical summaries and visualize data with histograms, boxplots, and scatter plots.
- **🤖 Machine Learning**:
  - Train models for **classification**, **regression**, and **clustering**.
  - Supported algorithms include Linear Regression, SGD Regressor, KNN, Decision Trees, Random Forest, and K-Means.
- **📈 Model Evaluation**: Analyze model performance with metrics like MSE, R², and confusion matrices.
- **📥 Downloadable Results**: Save generated plots and analysis results.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/alaaddinerenn/data-ml-interface.git
   cd data-ml-interface
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   streamlit run Home_Page.py
   ```

---

## Usage

1. **Home Page**: Navigate through the app's main sections.
2. **Data Analysis**: Upload your dataset and explore its structure and statistics.
3. **Machine Learning**: Train and evaluate models with your selected features and target variables.
4. **About Page**: Learn more about the app and its developer.

---

## Project Structure

```
data-ml-interface/
├── Home_Page.py                    # Main page
├── pages/                          # Streamlit pages
│   ├── 01_Analysis.py             
│   ├── 02_ML.py                   
│   └── 03_About.py                
├── models/                         # ML models (OOP architecture)
│   ├── __init__.py                # Public API exports
│   ├── base_classifier.py         # Base class for classifiers
│   ├── base_regressor.py          # Base class for regressors
│   ├── base_clusterer.py          # Base class for clusterers
│   ├── classifiers.py             # Decision Tree, KNN, Random Forest, XGBoost
│   ├── regressors.py              # Linear Regression, SGD Regressor
│   ├── clusterers.py              # K-Means, DBSCAN, Agglomerative
│   ├── classification_utils.py    # Classification analysis tools
│   ├── regression_utils.py        # Regression analysis tools
│   ├── clustering_utils.py        # Clustering analysis tools
│   └── utils.py                   # Model utilities
├── utils.py                        # Core utilities 
├── plotting.py                     # Visualization utilities
├── stats.py                        # Statistical analysis 
├── file.py                         # File management (FileManager class)
├── README.md                       # Project documentation
└── requirements.txt                # Python dependencies
```

**Architecture**: Object-Oriented Programming (OOP) with SOLID principles
- Each utility is organized into dedicated classes with clear responsibilities
- Base classes for ML models enable easy extension and maintenance
- Clean separation between data processing, visualization, and ML logic

---

## Technologies Used

- **Python**: Core programming language.
- **Streamlit**: Framework for building interactive web applications.
- **Pandas**: Data manipulation and analysis.
- **Scikit-learn**: Machine learning algorithms and preprocessing.
- **Matplotlib & Seaborn**: Data visualization.

---

## Future Enhancements

- Add support for additional file formats (e.g., JSON, Parquet).
- Implement advanced machine learning algorithms (e.g., Neural Networks, SVMs).
- Enhance visualization options with interactive plots.
- Add automated hyperparameter tuning for models.

---

## Developer

- **Alaaddin Eren Namlı**

Feel free to reach out for feedback or collaboration opportunities!
