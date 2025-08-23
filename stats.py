import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from plotting import plot_features_in_rows, plot_classhist, plot_boxplot, plot_hist
from utils import download_plot

def show_stats(df=None) -> None:
    if df is None:
        if "df_clean" in st.session_state and not st.session_state.df_clean.empty:
            df = st.session_state.df_clean
        elif "df" in st.session_state and not st.session_state.df.empty:
            df = st.session_state.df
    
    if df is not None and not df.empty:
        st.subheader("Yüklemiş Olduğunuz Veri")
        st.dataframe(df)

        st.subheader("İstatistiksel Özet")
        data_type = None
                
        # Hedef sütunlarını belirle
        target_cols = []

        if 'label' in df.columns:
            target_cols = ['label']
            data_type = 0  # classification
        elif 'target' in df.columns:
            target_cols = ['target']
            data_type = 1  # regression
        elif 'cluster' in df.columns:
            target_cols = ['cluster']
            data_type = 2  # cluster
        elif any(col.startswith("target") for col in df.columns):
            target_cols = [col for col in df.columns if col.startswith("target")]
            data_type = 3  # multiple regression

        # Özellik sütunları
        feature_df = df.drop(columns=target_cols)

        # Describe göster
        st.dataframe(feature_df.describe().round(3))

        if data_type == 0:  # CLASSIFICATION
            st.subheader("Sınıf Dağılımı")

            class_counts = df['label'].value_counts()

            fig, ax = plt.subplots()

            # autopct ile hem sayı hem yüzde yazalım
            def func(pct, allvals):
                absolute = int(round(pct/100.*sum(allvals)))
                return f"{absolute} ({pct:.1f}%)"

            wedges, texts, autotexts = ax.pie(
                class_counts,
                labels=class_counts.index,
                autopct=lambda pct: func(pct, class_counts),
                startangle=90
            )

            ax.axis("equal")
            st.pyplot(fig)
            download_plot(fig, "pie")


            df_info = pd.DataFrame({
                "Sütun Adı": list(df.columns),
                "Veri Tipi": [str(dt) for dt in df.dtypes]
            })

            st.subheader("Sütunlar ve Tipleri")
            st.dataframe(df_info)

            st.subheader("🔹 Hangi Özelliklerin Grafiklerini Görmek İstersin?")
            selected_features = st.multiselect(
                "Özellik(leri) seç:",
                options=df.columns[:-1],
                default=df.columns[:2],
                key="graph_4option"
            )

            st.subheader("🔹 Hangi Grafik Türlerini Gösterelim?")
            show_hist      = st.checkbox("📊 Histogram + KDE", value=True)
            show_classhist = st.checkbox("🎨 Sınıfa Göre Histogram", value=True)
            show_boxplot   = st.checkbox("📦 Boxplot", value=True)
            show_scatter   = st.checkbox("🔵 Scatter Plot", value=True)

            if show_hist:
                st.markdown("## 📊 Histogram + KDE")
                plot_features_in_rows(df, selected_features, plot_hist, n_cols=2)

            if show_classhist:
                st.markdown("## 🎨 Sınıfa Göre Histogram")
                plot_features_in_rows(df, selected_features, plot_classhist, n_cols=2)

            if show_boxplot:
                st.markdown("## 📦 Boxplot (Sınıfa Göre)")
                plot_features_in_rows(df, selected_features, plot_boxplot, n_cols=2)

            if show_scatter:
                st.markdown("## 🔵 Scatter Plot (2 Özellik Seçin)")
                scatter_features = st.multiselect(
                    "Scatter için iki özellik seçin:",
                    options=df.columns[:-1],
                    default=df.columns[:2],
                    key="scatter_features"
                )
                if len(scatter_features) == 2:
                    fig, ax = plt.subplots()
                    sns.scatterplot(
                        data=df,
                        x=scatter_features[0],
                        y=scatter_features[1],
                        hue='label',
                        style='label',
                        palette='Set2',
                        s=70,
                        alpha=0.8,
                        ax=ax
                    )
                    ax.set_title(f"Scatter Plot: {scatter_features[0]} vs {scatter_features[1]}")
                    st.pyplot(fig)
                    download_plot(fig, "classhist", scatter_features)  
                elif len(scatter_features) > 0:
                    st.warning("Lütfen scatter plot için **tam olarak iki özellik** seçin.")

        if data_type == 2:  # CLUSTERING
            st.subheader("Sınıf Dağılımı")

            class_counts = df['cluster'].value_counts()

            fig, ax = plt.subplots()

            # autopct ile hem sayı hem yüzde yazalım
            def func(pct, allvals):
                absolute = int(round(pct/100.*sum(allvals)))
                return f"{absolute} ({pct:.1f}%)"

            wedges, texts, autotexts = ax.pie(
                class_counts,
                labels=class_counts.index,
                autopct=lambda pct: func(pct, class_counts),
                startangle=90
            )

            ax.axis("equal")
            st.pyplot(fig)
            download_plot(fig, "pie")


            df_info = pd.DataFrame({
                "Sütun Adı": list(df.columns),
                "Veri Tipi": [str(dt) for dt in df.dtypes]
            })

            st.subheader("Sütunlar ve Tipleri")
            st.dataframe(df_info)

            st.subheader("🔹 Hangi Özelliklerin Grafiklerini Görmek İstersin?")
            selected_features = st.multiselect(
                "Özellik(leri) seç:",
                options=df.columns[:-1],
                default=df.columns[:2],
                key="graph_4option"
            )

            st.subheader("🔹 Hangi Grafik Türlerini Gösterelim?")
            show_hist      = st.checkbox("📊 Histogram + KDE", value=True)
            show_classhist = st.checkbox("🎨 Sınıfa Göre Histogram", value=True)
            show_boxplot   = st.checkbox("📦 Boxplot", value=True)
            show_scatter   = st.checkbox("🔵 Scatter Plot", value=True)

            if show_hist:
                st.markdown("## 📊 Histogram + KDE")
                plot_features_in_rows(df, selected_features, plot_hist, n_cols=2)

            if show_classhist:
                st.markdown("## 🎨 Sınıfa Göre Histogram")
                plot_features_in_rows(df, selected_features, plot_classhist, n_cols=2, hue="cluster")

            if show_boxplot:
                st.markdown("## 📦 Boxplot (Sınıfa Göre)")
                plot_features_in_rows(df, selected_features, plot_boxplot, n_cols=2, hue="cluster")

            if show_scatter:
                st.markdown("## 🔵 Scatter Plot (2 Özellik Seçin)")
                scatter_features = st.multiselect(
                    "Scatter için iki özellik seçin:",
                    options=df.columns[:-1],
                    default=df.columns[:2],
                    key="scatter_features"
                )
                if len(scatter_features) == 2:
                    fig, ax = plt.subplots()
                    sns.scatterplot(
                        data=df,
                        x=scatter_features[0],
                        y=scatter_features[1],
                        hue='cluster',
                        style='cluster',
                        palette='Set2',
                        s=70,
                        alpha=0.8,
                        ax=ax
                    )
                    ax.set_title(f"Scatter Plot: {scatter_features[0]} vs {scatter_features[1]}")
                    st.pyplot(fig)
                    download_plot(fig, "classhist", scatter_features)  
                elif len(scatter_features) > 0:
                    st.warning("Lütfen scatter plot için **tam olarak iki özellik** seçin.")
                        

            st.markdown("---")
