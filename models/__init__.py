"""
Models package for ML classifiers, regressors, and clusterers.

This package provides a clean, object-oriented interface for machine learning models.
Each model type (classifier, regressor, clusterer) has its own base class and 
concrete implementations.

Example:
    >>> from models import decision_tree_page, kmeans_page
    >>> decision_tree_page(df)  # Train decision tree on dataframe
    >>> kmeans_page(df)         # Cluster dataframe with k-means
"""

# Classifiers
from models.classifiers import (
    DecisionTreeModel,
    KNNModel,
    RandomForestModel,
    XGBoostModel,
    decision_tree_page,
    knn_page,
    random_forest_page,
    xgboost_classifier_page
)

# Regressors
from models.regressors import (
    LinearRegressionModel,
    SGDRegressorModel,
    linear_regression_page,
    sgd_regression_page
)

# Clusterers
from models.clusterers import (
    KMeansModel,
    DBSCANModel,
    AgglomerativeModel,
    kmeans_page,
    dbscan_page,
    agglomerative_page
)

__version__ = "1.0.0"

__all__ = [
    # Classifier classes
    'DecisionTreeModel',
    'KNNModel',
    'RandomForestModel',
    'XGBoostModel',
    # Classifier pages
    'decision_tree_page',
    'knn_page',
    'random_forest_page',
    'xgboost_classifier_page',
    
    # Regressor classes
    'LinearRegressionModel',
    'SGDRegressorModel',
    # Regressor pages
    'linear_regression_page',
    'sgd_regression_page',
    
    # Clusterer classes
    'KMeansModel',
    'DBSCANModel',
    'AgglomerativeModel',
    # Clusterer pages
    'kmeans_page',
    'dbscan_page',
    'agglomerative_page'
]
