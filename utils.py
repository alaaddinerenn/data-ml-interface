import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
from sklearn.preprocessing import LabelEncoder


def download_plot(fig, graph_type, feature_names=None, ext_default="jpg"):
    # Feature isimlerini dosya adı için düzenle
    if feature_names is None:
        feature_part = ""
    elif isinstance(feature_names, list):
        feature_part = "_vs_".join(feature_names)
    else:
        feature_part = str(feature_names)

    # 🚀 boşlukları '_' yap
    feature_part = feature_part.replace(" ", "_")

    formats = ["jpg", "png", "pdf"]

    # feature_part boşsa sadece graph_type kullan
    key_suffix = feature_part if feature_part else "default"

    format_key = f"format_{graph_type}_{key_suffix}"
    dl_key = f"dl_{graph_type}_{key_suffix}"

    if format_key not in st.session_state:
        st.session_state[format_key] = ext_default

    # YAN YANA düzen
    col_empty1, col1, col2, col_empty2 = st.columns([1, 2, 2, 1])  # dışta boş kolonlar

    with col1:
        format_sec = st.selectbox(
            "Format",
            formats,
            key=format_key,
            index=formats.index(st.session_state[format_key]),
            label_visibility="collapsed"
        )

    buf = io.BytesIO()
    fig.savefig(buf, format=format_sec, bbox_inches="tight")
    buf.seek(0)

    filename = f"{graph_type}{'_' + feature_part if feature_part else ''}.{format_sec}"

    mime_map = {"jpg": "image/jpeg", "png": "image/png", "pdf": "application/pdf"}

    with col2:
        st.download_button(
            label="⬇️",
            data=buf,
            file_name=filename,
            mime=mime_map[format_sec],
            key=dl_key,
            use_container_width=True
        )



def compare(df_before: pd.DataFrame, df_after: pd.DataFrame):
    st.subheader("🔍 Veri Temizleme Öncesi ve Sonrası Karşılaştırması")

    tab_num, tab_cat = st.tabs(["📊 Sayısal Veriler", "📋 Kategorik Veriler"])

    target_cols = [c for c in df_before.columns if c in ['label', 'target'] or c.startswith('target')]
    dfb = df_before.drop(columns=target_cols, errors='ignore')
    dfa = df_after.drop(columns=target_cols, errors='ignore')

    numeric_cols = [c for c in dfb.columns if pd.api.types.is_numeric_dtype(dfb[c])]
    categorical_cols = [c for c in dfb.columns if c not in numeric_cols]

    # --- SAYISAL TAB ---
    with tab_num:
        st.markdown("### 🧱 Eksik Değerler")
        na_before_num = dfb[numeric_cols].isna().sum()
        na_after_num = dfa[numeric_cols].isna().sum()
        na_df_num = pd.DataFrame({
            "Önce": na_before_num,
            "Sonra": na_after_num
        })
        st.dataframe(na_df_num)
        st.bar_chart(na_df_num)

        st.markdown("### 📏 Temel İstatistiksel Özet")
        if len(numeric_cols) == 0:
            st.info("Sayısal sütun bulunamadı.")
        else:
            selected_num_cols = st.multiselect("İstatistiksel özet için sayısal sütun seç", numeric_cols, default=numeric_cols[:5])
            if selected_num_cols:
                st.markdown("#### Önce")
                st.dataframe(dfb[selected_num_cols].describe().round(3))
                st.markdown("#### Sonra")
                st.dataframe(dfa[selected_num_cols].describe().round(3))

        st.markdown("### 📉 Histogram Karşılaştırması")
        if len(numeric_cols) == 0:
            st.info("Sayısal sütun bulunamadı.")
        else:
            selected_hist_cols = st.multiselect("Histogram için sayısal sütun seç", numeric_cols, default=numeric_cols[:3], key="hist_num_cols")
            for col in selected_hist_cols:
                fig, axes = plt.subplots(1,2, figsize=(10,4))
                sns.histplot(dfb[col].dropna(), ax=axes[0], color='orange', kde=True)
                axes[0].set_title(f"Önce: {col}")
                sns.histplot(dfa[col].dropna(), ax=axes[1], color='blue', kde=True)
                axes[1].set_title(f"Sonra: {col}")
                st.pyplot(fig)

    # --- KATEGORİK TAB ---
    with tab_cat:
        st.markdown("### 🧱 Eksik Değerler")
        na_before_cat = dfb[categorical_cols].isna().sum()
        na_after_cat = dfa[categorical_cols].isna().sum()
        na_df_cat = pd.DataFrame({
            "Önce": na_before_cat,
            "Sonra": na_after_cat
        })
        st.dataframe(na_df_cat)
        st.bar_chart(na_df_cat)

        if len(categorical_cols) == 0:
            st.info("Kategorik sütun bulunamadı.")
            return

        selected_cat_cols = st.multiselect("Kategorik sütun seç", categorical_cols, default=categorical_cols[:3], key="cat_cols")

        for col in selected_cat_cols:
            st.markdown(f"## {col}")

            # Benzersiz değer sayısı
            unique_before = dfb[col].nunique(dropna=False)
            unique_after = dfa[col].nunique(dropna=False)
            st.markdown(f"**Benzersiz Değer Sayısı:** Önce: {unique_before} | Sonra: {unique_after}")

            # En çok görülen kategoriler (%)
            freq_before_pct = dfb[col].fillna("NaN").value_counts(normalize=True).head(5) * 100
            freq_after_pct = dfa[col].fillna("NaN").value_counts(normalize=True).head(5) * 100

            all_categories = freq_before_pct.index.union(freq_after_pct.index)

            freq_before_pct = freq_before_pct.reindex(all_categories).fillna(0)
            freq_after_pct = freq_after_pct.reindex(all_categories).fillna(0)

            # Tüm before verisinden oranları al (tüm kategoriler)
            full_before_pct = dfb[col].fillna("NaN").value_counts(normalize=True) * 100

            for cat in all_categories:
                if freq_before_pct[cat] == 0 and freq_after_pct[cat] > 0:
                    freq_before_pct[cat] = full_before_pct.get(cat, 0)

            freq_pct_df = pd.DataFrame({
                "Önce (%)": freq_before_pct,
                "Sonra (%)": freq_after_pct
            }).fillna(0)

            freq_pct_df = freq_pct_df.map(lambda x: f"{x:.2f}%")

            st.markdown("**En Çok Görülen Kategoriler (Yüzde %)**")
            st.dataframe(freq_pct_df)

            
            # Frekans karşılaştırması (adet)
            top_n = 10

            # Temizlenmemiş ve temizlenmiş veriden ilk 10 kategori
            before_top = dfb[col].fillna("NaN").value_counts().head(top_n).index
            after_top = dfa[col].fillna("NaN").value_counts().head(top_n).index

            # Birleşik kategori listesi (farklı kategoriler de dahil)
            all_top_categories = sorted(set(before_top) | set(after_top))

            # Orijinal (full) frekanslar
            freq_before_full = dfb[col].fillna("NaN").value_counts().reindex(all_top_categories, fill_value=0)
            freq_after_full = dfa[col].fillna("NaN").value_counts().reindex(all_top_categories, fill_value=0)

            # Başlangıçta top kategoriler için sadece ilk 10’lardaki frekanslar
            freq_before_top = dfb[col].fillna("NaN").value_counts().head(top_n).reindex(all_top_categories, fill_value=0)
            freq_after_top = dfa[col].fillna("NaN").value_counts().head(top_n).reindex(all_top_categories, fill_value=0)

            # 0 olan kategoriler için, orijinal frekansları kullan
            for cat in all_top_categories:
                if freq_before_top[cat] == 0 and freq_before_full[cat] > 0:
                    freq_before_top[cat] = freq_before_full[cat]
                if freq_after_top[cat] == 0 and freq_after_full[cat] > 0:
                    freq_after_top[cat] = freq_after_full[cat]

            # Kalan kategorileri "Diğer" olarak toplayalım (top_n dışındaki kategoriler)
            others_before = dfb[col].fillna("NaN").value_counts().drop(index=all_top_categories, errors='ignore').sum()
            others_after = dfa[col].fillna("NaN").value_counts().drop(index=all_top_categories, errors='ignore').sum()

            freq_before_top = pd.concat([freq_before_top, pd.Series({"Diğer": others_before})])
            freq_after_top = pd.concat([freq_after_top, pd.Series({"Diğer": others_after})])

            freq_df = pd.DataFrame({
                "Önce": freq_before_top,
                "Sonra": freq_after_top
            }).astype(int)

            st.markdown(f"**Frekans Dağılımı (Temizlenmemiş İlk {top_n} + Temizlenmiş İlk {top_n} + Diğer)**")
            st.dataframe(freq_df)

            # Bar chart – düzgün sıralı
            freq_df_sorted = freq_df.sort_values("Önce", ascending=False)

            fig, ax = plt.subplots(figsize=(10, 4))
            x = np.arange(len(freq_df_sorted.index))
            bar_width = 0.35

            ax.bar(x - bar_width/2, freq_df_sorted["Önce"], width=bar_width, label="Önce", color='orange')
            ax.bar(x + bar_width/2, freq_df_sorted["Sonra"], width=bar_width, label="Sonra", color='blue')

            ax.set_xticks(x)
            ax.set_xticklabels(freq_df_sorted.index, rotation=45, ha='right')
            ax.set_ylabel("Frekans")
            ax.set_title(f"{col} - Kategorik Değer Frekans Karşılaştırması")
            ax.legend()

            st.pyplot(fig)

            # Modal değer karşılaştırması
            mode_before = dfb[col].mode().iloc[0] if not dfb[col].mode().empty else "Yok"
            mode_after = dfa[col].mode().iloc[0] if not dfa[col].mode().empty else "Yok"
            st.markdown(f"**En Sık Değer (Mod):** Önce: {mode_before} | Sonra: {mode_after}")

            # Kategorilerde değişiklik kontrolü
            set_before = set(dfb[col].dropna().unique())
            set_after = set(dfa[col].dropna().unique())
            added = set_after - set_before
            removed = set_before - set_after
            st.markdown(f"**Yeni Eklenen Kategoriler:** {added if added else 'Yok'}")
            st.markdown(f"**Kaldırılan Kategoriler:** {removed if removed else 'Yok'}")

def clean_data(df):
    if df is None or df.empty:
        st.warning("⚠️ Veri yok.")
        return df, False

    missing_values = ["None", "NA", "Missing", "?", "", "na", "NaN", "N/A", "n/a"]
    df_cleaned = df.copy()
    df_cleaned.replace(missing_values, pd.NA, inplace=True)

    na_cols = df_cleaned.columns[df_cleaned.isna().any()].tolist()

    # 📌 Eğer hiç eksik veri yoksa buton göstermeden çık
    if not na_cols:
        st.info("✅ Eksik değer bulunmamaktadır, temizleme işlemi yapılmadı.")
        return df_cleaned, True, True
    
    actions = {}
    
    st.subheader("🧹 Eksik Değer Temizleme")
    
    for col in na_cols:
        st.markdown(f"**{col}** sütununda `{df_cleaned[col].isna().sum()}` eksik değer var.")
        
        is_numeric = pd.api.types.is_numeric_dtype(df_cleaned[col])
        if is_numeric:
            options = ["Hiçbir şey yapma", "Ortalama ile doldur", "Medyan ile doldur", "Mod ile doldur", "Satırları sil"]
        else:
            options = ["Hiçbir şey yapma", "Mod ile doldur", "Satırları sil"]

        action = st.selectbox(
            f"{col} sütunu için ne yapılsın?",
            options,
            key=f"na_action_{col}"
        )
        actions[col] = action

    if st.button("🧹 Temizlemeyi Uygula"):
        for col, action in actions.items():
            if action == "Ortalama ile doldur":
                df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mean())
            elif action == "Medyan ile doldur":
                df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())
            elif action == "Mod ile doldur":
                mode_val = df_cleaned[col].mode()
                if not mode_val.empty:
                    df_cleaned[col] = df_cleaned[col].fillna(mode_val[0])
                else:
                    st.warning(f"⚠️ {col} sütunu için mod değeri bulunamadı.")
            elif action == "Satırları sil":
                df_cleaned = df_cleaned[df_cleaned[col].notna()]
        
        st.success("✅ Eksik değerler temizlendi.")
        return df_cleaned, True, False

    return df, False, False

# Temizleme öncesi Ukraine yüzdesi: %4.0833
# Temizleme sonrası Ukraine yüzdesi: %6.2897

def encode_features(df, encoding_type="One-Hot Encoding"):
    df_encoded = df.copy()
    if encoding_type == "One-Hot Encoding":
        df_encoded = pd.get_dummies(df_encoded)
    elif encoding_type == "Label Encoding":
        le = LabelEncoder()
        for col in df_encoded.select_dtypes(include=['object']).columns:
            df_encoded[col] = le.fit_transform(df_encoded[col])
    return df_encoded