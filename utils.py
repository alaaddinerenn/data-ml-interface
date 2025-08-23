import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
from sklearn.preprocessing import LabelEncoder


def download_plot(fig, graph_type, feature_names=None, ext_default="jpg"):
    # Format feature names for the file name
    if feature_names is None:
        feature_part = ""
    elif isinstance(feature_names, list):
        feature_part = "_vs_".join(feature_names)
    else:
        feature_part = str(feature_names)

    # Replace spaces with '_'
    feature_part = feature_part.replace(" ", "_")

    formats = ["jpg", "png", "pdf"]

    # Use only graph_type if feature_part is empty
    key_suffix = feature_part if feature_part else "default"

    format_key = f"format_{graph_type}_{key_suffix}"
    dl_key = f"dl_{graph_type}_{key_suffix}"

    if format_key not in st.session_state:
        st.session_state[format_key] = ext_default

    # Layout side by side
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
    st.subheader("🔍 Comparison Before and After Data Cleaning")

    tab_num, tab_cat = st.tabs(["📊 Numerical Data", "📋 Categorical Data"])

    target_cols = [c for c in df_before.columns if c in ['label', 'target'] or c.startswith('target')]
    dfb = df_before.drop(columns=target_cols, errors='ignore')
    dfa = df_after.drop(columns=target_cols, errors='ignore')

    numeric_cols = [c for c in dfb.columns if pd.api.types.is_numeric_dtype(dfb[c])]
    categorical_cols = [c for c in dfb.columns if c not in numeric_cols]

    # --- NUMERICAL TAB ---
    with tab_num:
        st.markdown("### 🧱 Missing Values")
        na_before_num = dfb[numeric_cols].isna().sum()
        na_after_num = dfa[numeric_cols].isna().sum()
        na_df_num = pd.DataFrame({
            "Before": na_before_num,
            "After": na_after_num
        })
        st.dataframe(na_df_num)
        st.bar_chart(na_df_num)

        st.markdown("### 📏 Basic Statistical Summary")
        if len(numeric_cols) == 0:
            st.info("No numerical columns found.")
        else:
            selected_num_cols = st.multiselect("Select numerical columns for summary", numeric_cols, default=numeric_cols[:5])
            if selected_num_cols:
                st.markdown("#### Before")
                st.dataframe(dfb[selected_num_cols].describe().round(3))
                st.markdown("#### After")
                st.dataframe(dfa[selected_num_cols].describe().round(3))

        st.markdown("### 📉 Histogram Comparison")
        if len(numeric_cols) == 0:
            st.info("No numerical columns found.")
        else:
            selected_hist_cols = st.multiselect("Select numerical columns for histogram", numeric_cols, default=numeric_cols[:3], key="hist_num_cols")
            for col in selected_hist_cols:
                fig, axes = plt.subplots(1,2, figsize=(10,4))
                sns.histplot(dfb[col].dropna(), ax=axes[0], color='orange', kde=True)
                axes[0].set_title(f"Before: {col}")
                sns.histplot(dfa[col].dropna(), ax=axes[1], color='blue', kde=True)
                axes[1].set_title(f"After: {col}")
                st.pyplot(fig)

    # --- CATEGORICAL TAB ---
    with tab_cat:
        st.markdown("### 🧱 Missing Values")
        na_before_cat = dfb[categorical_cols].isna().sum()
        na_after_cat = dfa[categorical_cols].isna().sum()
        na_df_cat = pd.DataFrame({
            "Before": na_before_cat,
            "After": na_after_cat
        })
        st.dataframe(na_df_cat)
        st.bar_chart(na_df_cat)

        if len(categorical_cols) == 0:
            st.info("No categorical columns found.")
            return

        selected_cat_cols = st.multiselect("Select categorical columns", categorical_cols, default=categorical_cols[:3], key="cat_cols")

        for col in selected_cat_cols:
            st.markdown(f"## {col}")

            # Unique value count
            unique_before = dfb[col].nunique(dropna=False)
            unique_after = dfa[col].nunique(dropna=False)
            st.markdown(f"**Unique Values:** Before: {unique_before} | After: {unique_after}")

            # Most frequent categories (%)
            freq_before_pct = dfb[col].fillna("NaN").value_counts(normalize=True).head(5) * 100
            freq_after_pct = dfa[col].fillna("NaN").value_counts(normalize=True).head(5) * 100

            all_categories = freq_before_pct.index.union(freq_after_pct.index)

            freq_before_pct = freq_before_pct.reindex(all_categories).fillna(0)
            freq_after_pct = freq_after_pct.reindex(all_categories).fillna(0)

            # Use full dataset percentages for categories with 0 frequency
            full_before_pct = dfb[col].fillna("NaN").value_counts(normalize=True) * 100

            for cat in all_categories:
                if freq_before_pct[cat] == 0 and freq_after_pct[cat] > 0:
                    freq_before_pct[cat] = full_before_pct.get(cat, 0)

            freq_pct_df = pd.DataFrame({
                "Before (%)": freq_before_pct,
                "After (%)": freq_after_pct
            }).fillna(0)

            freq_pct_df = freq_pct_df.map(lambda x: f"{x:.2f}%")

            st.markdown("**Most Frequent Categories (Percentage %)**")
            st.dataframe(freq_pct_df)

            # Frequency comparison (count)
            top_n = 10

            # Top categories from uncleaned and cleaned data
            before_top = dfb[col].fillna("NaN").value_counts().head(top_n).index
            after_top = dfa[col].fillna("NaN").value_counts().head(top_n).index

            # Combined category list (including different categories)
            all_top_categories = sorted(set(before_top) | set(after_top))

            # Original (full) frequencies
            freq_before_full = dfb[col].fillna("NaN").value_counts().reindex(all_top_categories, fill_value=0)
            freq_after_full = dfa[col].fillna("NaN").value_counts().reindex(all_top_categories, fill_value=0)

            # Initial frequencies for top categories
            freq_before_top = dfb[col].fillna("NaN").value_counts().head(top_n).reindex(all_top_categories, fill_value=0)
            freq_after_top = dfa[col].fillna("NaN").value_counts().head(top_n).reindex(all_top_categories, fill_value=0)

            # Use original frequencies for categories with 0 frequency
            for cat in all_top_categories:
                if freq_before_top[cat] == 0 and freq_before_full[cat] > 0:
                    freq_before_top[cat] = freq_before_full[cat]
                if freq_after_top[cat] == 0 and freq_after_full[cat] > 0:
                    freq_after_top[cat] = freq_after_full[cat]

            # Combine remaining categories as "Others"
            others_before = dfb[col].fillna("NaN").value_counts().drop(index=all_top_categories, errors='ignore').sum()
            others_after = dfa[col].fillna("NaN").value_counts().drop(index=all_top_categories, errors='ignore').sum()

            freq_before_top = pd.concat([freq_before_top, pd.Series({"Others": others_before})])
            freq_after_top = pd.concat([freq_after_top, pd.Series({"Others": others_after})])

            freq_df = pd.DataFrame({
                "Before": freq_before_top,
                "After": freq_after_top
            }).astype(int)

            st.markdown(f"**Frequency Distribution (Uncleaned Top {top_n} + Cleaned Top {top_n} + Others)**")
            st.dataframe(freq_df)

            # Bar chart – properly sorted
            freq_df_sorted = freq_df.sort_values("Before", ascending=False)

            fig, ax = plt.subplots(figsize=(10, 4))
            x = np.arange(len(freq_df_sorted.index))
            bar_width = 0.35

            ax.bar(x - bar_width/2, freq_df_sorted["Before"], width=bar_width, label="Before", color='orange')
            ax.bar(x + bar_width/2, freq_df_sorted["After"], width=bar_width, label="After", color='blue')

            ax.set_xticks(x)
            ax.set_xticklabels(freq_df_sorted.index, rotation=45, ha='right')
            ax.set_ylabel("Frequency")
            ax.set_title(f"{col} - Categorical Value Frequency Comparison")
            ax.legend()

            st.pyplot(fig)

            # Mode value comparison
            mode_before = dfb[col].mode().iloc[0] if not dfb[col].mode().empty else "None"
            mode_after = dfa[col].mode().iloc[0] if not dfa[col].mode().empty else "None"
            st.markdown(f"**Most Frequent Value (Mode):** Before: {mode_before} | After: {mode_after}")

            # Check for changes in categories
            set_before = set(dfb[col].dropna().unique())
            set_after = set(dfa[col].dropna().unique())
            added = set_after - set_before
            removed = set_before - set_after
            st.markdown(f"**Newly Added Categories:** {added if added else 'None'}")
            st.markdown(f"**Removed Categories:** {removed if removed else 'None'}")

def clean_data(df):
    if df is None or df.empty:
        st.warning("⚠️ No data available.")
        return df, False

    missing_values = ["None", "NA", "Missing", "?", "", "na", "NaN", "N/A", "n/a"]
    df_cleaned = df.copy()
    df_cleaned.replace(missing_values, pd.NA, inplace=True)

    na_cols = df_cleaned.columns[df_cleaned.isna().any()].tolist()

    # 📌 If no missing data, exit without showing the button
    if not na_cols:
        st.info("✅ No missing values found, no cleaning performed.")
        return df_cleaned, True, True
    
    actions = {}
    
    st.subheader("🧹 Missing Value Cleaning")
    
    for col in na_cols:
        st.markdown(f"**{col}** column has `{df_cleaned[col].isna().sum()}` missing values.")
        
        is_numeric = pd.api.types.is_numeric_dtype(df_cleaned[col])
        if is_numeric:
            options = ["Do nothing", "Fill with mean", "Fill with median", "Fill with mode", "Drop rows"]
        else:
            options = ["Do nothing", "Fill with mode", "Drop rows"]

        action = st.selectbox(
            f"What should be done for the {col} column?",
            options,
            key=f"na_action_{col}"
        )
        actions[col] = action

    if st.button("🧹 Apply Cleaning"):
        for col, action in actions.items():
            if action == "Fill with mean":
                df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mean())
            elif action == "Fill with median":
                df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())
            elif action == "Fill with mode":
                mode_val = df_cleaned[col].mode()
                if not mode_val.empty:
                    df_cleaned[col] = df_cleaned[col].fillna(mode_val[0])
                else:
                    st.warning(f"⚠️ No mode value found for the {col} column.")
            elif action == "Drop rows":
                df_cleaned = df_cleaned[df_cleaned[col].notna()]
        
        st.success("✅ Missing values cleaned.")
        return df_cleaned, True, False

    return df, False, False

def encode_features(df, encoding_type="One-Hot Encoding"):
    df_encoded = df.copy()
    if encoding_type == "One-Hot Encoding":
        df_encoded = pd.get_dummies(df_encoded)
    elif encoding_type == "Label Encoding":
        le = LabelEncoder()
        for col in df_encoded.select_dtypes(include=['object']).columns:
            df_encoded[col] = le.fit_transform(df_encoded[col])
    return df_encoded