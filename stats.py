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
        st.subheader("Uploaded Data")
        st.dataframe(df)

        st.subheader("Statistical Summary")
        data_type = None

        # Determine target columns
        target_cols = []

        if 'label' in df.columns:
            target_cols = ['label']
            data_type = 0  # classification
        elif 'target' in df.columns:
            target_cols = ['target']
            data_type = 1  # regression
        elif 'cluster' in df.columns:
            target_cols = ['cluster']
            data_type = 2  # clustering
        elif any(col.startswith("target") for col in df.columns):
            target_cols = [col for col in df.columns if col.startswith("target")]
            data_type = 3  # multiple regression

        # Feature columns
        feature_df = df.drop(columns=target_cols)

        # Display describe
        st.dataframe(feature_df.describe().round(3))

        if data_type == 0:  # CLASSIFICATION
            st.subheader("Class Distribution")

            class_counts = df['label'].value_counts()

            fig, ax = plt.subplots()

            # Display both count and percentage in pie chart
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
                "Column Name": list(df.columns),
                "Data Type": [str(dt) for dt in df.dtypes]
            })

            st.subheader("Columns and Data Types")
            st.dataframe(df_info)

            st.subheader("🔹 Which Features Would You Like to Visualize?")
            selected_features = st.multiselect(
                "Select feature(s):",
                options=df.columns[:-1],
                default=df.columns[:2],
                key="graph_4option"
            )

            st.subheader("🔹 Which Plot Types Should We Show?")
            show_hist      = st.checkbox("📊 Histogram + KDE", value=True)
            show_classhist = st.checkbox("🎨 Histogram by Class", value=True)
            show_boxplot   = st.checkbox("📦 Boxplot", value=True)
            show_scatter   = st.checkbox("🔵 Scatter Plot", value=True)

            if show_hist:
                st.markdown("## 📊 Histogram + KDE")
                plot_features_in_rows(df, selected_features, plot_hist, n_cols=2)

            if show_classhist:
                st.markdown("## 🎨 Histogram by Class")
                plot_features_in_rows(df, selected_features, plot_classhist, n_cols=2)

            if show_boxplot:
                st.markdown("## 📦 Boxplot (By Class)")
                plot_features_in_rows(df, selected_features, plot_boxplot, n_cols=2)

            if show_scatter:
                st.markdown("## 🔵 Scatter Plot (Select 2 Features)")
                scatter_features = st.multiselect(
                    "Select two features for scatter plot:",
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
                    st.warning("Please select exactly two features for the scatter plot.")

        if data_type == 2:  # CLUSTERING
            st.subheader("Cluster Distribution")

            class_counts = df['cluster'].value_counts()

            fig, ax = plt.subplots()

            # Display both count and percentage in pie chart
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
                "Column Name": list(df.columns),
                "Data Type": [str(dt) for dt in df.dtypes]
            })

            st.subheader("Columns and Data Types")
            st.dataframe(df_info)

            st.subheader("🔹 Which Features Would You Like to Visualize?")
            selected_features = st.multiselect(
                "Select feature(s):",
                options=df.columns[:-1],
                default=df.columns[:2],
                key="graph_4option"
            )

            st.subheader("🔹 Which Plot Types Should We Show?")
            show_hist      = st.checkbox("📊 Histogram + KDE", value=True)
            show_classhist = st.checkbox("🎨 Histogram by Class", value=True)
            show_boxplot   = st.checkbox("📦 Boxplot", value=True)
            show_scatter   = st.checkbox("🔵 Scatter Plot", value=True)

            if show_hist:
                st.markdown("## 📊 Histogram + KDE")
                plot_features_in_rows(df, selected_features, plot_hist, n_cols=2)

            if show_classhist:
                st.markdown("## 🎨 Histogram by Class")
                plot_features_in_rows(df, selected_features, plot_classhist, n_cols=2, hue="cluster")

            if show_boxplot:
                st.markdown("## 📦 Boxplot (By Class)")
                plot_features_in_rows(df, selected_features, plot_boxplot, n_cols=2, hue="cluster")

            if show_scatter:
                st.markdown("## 🔵 Scatter Plot (Select 2 Features)")
                scatter_features = st.multiselect(
                    "Select two features for scatter plot:",
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
                    st.warning("Please select exactly two features for the scatter plot.")

            st.markdown("---")
