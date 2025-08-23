import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pandas as pd
from utils import download_plot


def chunks(lst, n):
    """Listeyi n elemanlı parçalara böl."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

# Grafik çizme fonksiyonu, seçilen özellik listesi ve çizilecek grafik fonksiyonu parametre olarak alınabilir
def plot_features_in_rows(df, selected_features, plot_func, n_cols=2, hue="label"):
    for chunk_features in chunks(selected_features, n_cols):
        cols = st.columns(len(chunk_features))
        for i, feature in enumerate(chunk_features):
            with cols[i]:
                plot_func(feature, df, hue)

# Örnek: Histogram + KDE için çizim fonksiyonu
def plot_hist(feature, df, hue, palette=None) -> None:
    st.markdown(f"**{feature}**")
    fig, ax = plt.subplots()
    sns.histplot(data=df, x=feature, kde=True, bins=20, ax=ax)
    st.pyplot(fig)
    download_plot(fig, "hist", feature.strip())
    

# Örnek: Sınıfa Göre histogram
def plot_classhist(feature, df, hue, palette=None) -> None:
    st.markdown(f"**{feature}**")
    fig, ax = plt.subplots()
    sns.histplot(data=df, x=feature, hue=hue if hue in df.columns else None, kde=True, bins=20, ax=ax, palette=palette)
    st.pyplot(fig)
    download_plot(fig, "classhist", feature.strip())    

# Örnek: Boxplot
def plot_boxplot(feature, df, hue, palette=None) -> None:
    st.markdown(f"**{feature}**")
    fig, ax = plt.subplots()
    sns.boxplot(data=df, x=hue if hue in df.columns else None, hue=hue if hue in df.columns else None, y=feature, ax=ax, palette=palette)
    st.pyplot(fig)
    download_plot(fig, "boxplot", feature.strip())


