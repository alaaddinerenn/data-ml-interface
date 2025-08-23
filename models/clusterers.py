import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler
from utils import download_plot
from models.utils import encode_features


# --- Model Eğitimi ---
def train_kmeans(df) -> None:
    st.subheader("🔹 KMeans Ayarları")

    # Kullanılacak feature seçimi
    features = st.multiselect(
        "Clustering için kullanılacak özellikleri seçin",
        options=df.columns,
        default=[col for col in df.columns if col != "cluster"]
    )

    n_clusters = st.slider("Küme Sayısı (k)", 2, 10, 3)
    encoding_type = st.radio("Encoding Tipi", ["One-Hot Encoding", "Label Encoding"])
    scale_data = st.checkbox("Veriyi ölçeklendir (StandardScaler)", value=True)
    random_seed = st.number_input("Random Seed", value=42)

    if st.button("Modeli Eğit", key="btn_train_kmeans"):
        if not features:
            st.warning("En az bir özellik seçmelisiniz.")
            return

        # Encoding
        df_encoded = encode_features(df, encoding_type)

        # X matrisi
        X = df_encoded[features]

        # Scaling opsiyonel
        if scale_data:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
        else:
            X = X.values

        # Model
        model = KMeans(n_clusters=n_clusters, random_state=random_seed, n_init=10)
        y_pred = model.fit_predict(X)

        # Session’a kaydet
        st.session_state["kmeans_results"] = {
            "model": model,
            "X": X,
            "y_pred": y_pred,
            "features": features,
            "scaled": scale_data
        }

        st.success("✅ KMeans modeli başarıyla eğitildi!")


# --- Analiz ---
def kmeans_analysis() -> None:
    if "kmeans_results" not in st.session_state:
        st.warning("⚠ Önce modeli eğitmelisiniz.")
        return

    results = st.session_state["kmeans_results"]
    model = results["model"]
    X = results["X"]
    y_pred = results["y_pred"]

    st.subheader("📊 Analiz Araçları")
    analysis_options = st.multiselect(
        "Görselleştirmek istediğiniz analizleri seçin",
        ["Cluster Distribution", "Silhouette Score", "PCA 2D Plot", "Elbow Curve"],
        default=["Cluster Distribution"]
    )

    if "Cluster Distribution" in analysis_options:
        st.write("🔹 Küme dağılımı")
        unique, counts = np.unique(y_pred, return_counts=True)
        dist_df = pd.DataFrame({"Cluster": unique, "Count": counts})
        st.dataframe(dist_df)

        fig, ax = plt.subplots()

        # autopct ile hem sayı hem yüzde yazalım
        def func(pct, allvals):
            absolute = int(round(pct/100.*sum(allvals)))
            return f"{absolute} ({pct:.1f}%)"

        pie_colors = ["#1E90FF", "#32CD32", "#FFA500"]
        wedges, texts, autotexts = ax.pie(
            counts,
            labels=unique,
            autopct=lambda pct: func(pct, counts),
            startangle=90,
            colors=pie_colors[:len(unique)]
        )

        ax.axis("equal")  # Daireyi eşit yap
        st.pyplot(fig)
        download_plot(fig, "cluster_distribution_pie")

        # Boxplot: plotting.py fonksiyonu ile
        st.subheader("🔹 Kümelere Göre Özelliklerin Boxplot'u")
        from plotting import plot_boxplot
        features = results["features"]
        df_box = pd.DataFrame(X, columns=features)
        df_box["cluster"] = y_pred
        box_colors = ["#1E90FF", "#32CD32", "#FFA500"]
        for feature in features:
            plot_boxplot(feature, df_box, hue="cluster", palette=box_colors)

    if "Silhouette Score" in analysis_options:
        score = silhouette_score(X, y_pred)
        st.subheader(f"🔹 **Silhouette Score:** {score:.4f}")
        # Silhouette Plot (sklearn style)
        sample_scores = silhouette_samples(X, y_pred)
        n_clusters = len(np.unique(y_pred))
        sil_colors = ["#1E90FF", "#32CD32", "#FFA500"]
        fig, ax = plt.subplots(figsize=(8, 5))
        y_lower = 10
        for i in range(n_clusters):
            ith_cluster_silhouette_values = sample_scores[y_pred == i]
            ith_cluster_silhouette_values.sort()
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
            color = sil_colors[i % len(sil_colors)]
            ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values,
                             facecolor=color, edgecolor=color, alpha=0.7)
            ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            y_lower = y_upper + 10  # 10 for spacing between clusters
        ax.axvline(score, color="red", linestyle="--", label="Ortalama Silhouette")
        ax.set_xlabel("Silhouette Coefficient Values")
        ax.set_ylabel("Sample Index")
        ax.set_title("Silhouette Plot for Each Cluster")
        ax.legend()
        st.pyplot(fig)
        download_plot(fig, "silhouette_plot")
        
        # k vs silhouette score plot
        st.subheader("🔹 k vs Silhouette Score")
        k_range = range(2, 11)
        sil_scores = []
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X)
            sil_scores.append(silhouette_score(X, labels))
        fig2, ax2 = plt.subplots()
        ax2.plot(k_range, sil_scores, marker="o")
        ax2.set_xlabel("Küme Sayısı (k)")
        ax2.set_ylabel("Ortalama Silhouette Score")
        ax2.set_title("k vs Silhouette Score")
        st.pyplot(fig2)
        download_plot(fig2, "k_vs_silhouette_score")

    if "PCA 2D Plot" in analysis_options:
        st.subheader("🔹 PCA ile 2D Küme Görselleştirme")
        from matplotlib.colors import ListedColormap
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        fig, ax = plt.subplots()
        pca_colors = ["#1E90FF", "#32CD32", "#FFA500"]
        cmap = ListedColormap(pca_colors[:len(np.unique(y_pred))])
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y_pred, cmap=cmap, s=30)
        ax.set_xlabel("PCA1")
        ax.set_ylabel("PCA2")
        ax.set_title("Clusters (PCA 2D)")
        plt.colorbar(scatter, ax=ax)
        st.pyplot(fig)
        download_plot(fig, "pca_clusters")

    if "Elbow Curve" in analysis_options:
        st.subheader("🔹 Elbow Method")
        distortions = []
        K = range(2, 11)
        for k in K:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(X)
            distortions.append(km.inertia_)

        fig, ax = plt.subplots()
        ax.plot(K, distortions, "bo-")
        ax.set_xlabel("k")
        ax.set_ylabel("Inertia")
        ax.set_title("Elbow Curve")
        st.pyplot(fig)
        download_plot(fig, "elbow_curve")


# --- Page ---
def kmeans_page(df) -> None:
    st.title("KMeans Clustering")
    train_kmeans(df)
    st.markdown("---")
    kmeans_analysis()

