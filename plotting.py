import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pandas as pd
from utils import download_plot


def chunks(lst, n):
    """Split the list into chunks of size n."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


# Function to plot graphs, takes a list of selected features and a plotting function as parameters
def plot_features_in_rows(df, selected_features, plot_func, n_cols=2, hue="label"):
    for chunk_features in chunks(selected_features, n_cols):
        cols = st.columns(len(chunk_features))
        for i, feature in enumerate(chunk_features):
            with cols[i]:
                plot_func(feature, df, hue)


# Function to plot Histogram + KDE
def plot_hist(feature, df, hue, palette=None) -> None:
    st.markdown(f"**{feature}**")
    fig, ax = plt.subplots()
    sns.histplot(data=df, x=feature, kde=True, bins=20, ax=ax)
    st.pyplot(fig)
    download_plot(fig, "hist", feature.strip())


# Histogram by class
def plot_classhist(feature, df, hue, palette=None) -> None:
    st.markdown(f"**{feature}**")
    fig, ax = plt.subplots()
    sns.histplot(data=df, x=feature, hue=hue if hue in df.columns else None, kde=True, bins=20, ax=ax, palette=palette)
    st.pyplot(fig)
    download_plot(fig, "classhist", feature.strip())    
    

# Boxplot
def plot_boxplot(feature, df, hue, palette=None) -> None:
    st.markdown(f"**{feature}**")
    fig, ax = plt.subplots()
    sns.boxplot(data=df, x=hue if hue in df.columns else None, hue=hue if hue in df.columns else None, y=feature, ax=ax, palette=palette)
    st.pyplot(fig)
    download_plot(fig, "boxplot", feature.strip())


