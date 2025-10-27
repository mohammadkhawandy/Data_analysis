import sys
import os
import streamlit as st
import pandas as pd
import joblib
import numpy as np


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(file), ".."))
sys.path.insert(0, BASE_DIR)

from src.utils import preprocess, evaluate_model
from src.ga_selector import GAFeatureSelector

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Feature Selection App", page_icon="📊")

st.title(" تطبيق اختيار الميزات الذكي")
st.write("تستطيع رفع اي ملف بيانات وتحليله بعدة طرق لاختيار افضل الميزات")

uploaded_file = st.file_uploader("CSV ارفع ملف ", type=["csv"])

if uploaded_file is not None:

    #  Choose a data separator
    sep = st.radio(" اختر الفاصل بين الأعمدة:", {",": "Comma (,)", ";": "Semicolon (;)"})

    # Read file by selected separator
    df = pd.read_csv(uploaded_file, sep=sep)

    st.subheader(" معاينة البيانات")
    st.dataframe(df.head())

    target_col = st.selectbox("  (Target)اختر العمود الهدف", df.columns)

    if st.button(" تشغيل التحليل"):
        with st.spinner(" جاري المعالجة..."):
            try:
                X = df.drop(columns=[target_col])
                y = df[target_col]

                X_processed = pd.get_dummies(X, drop_first=True)
                if y.dtype == "object":
                    y = y.astype("category").cat.codes

                st.success(f" شكل البيانات بعد المعالجة: {X_processed.shape}")

                results = {}
                clf = LogisticRegression(max_iter=500)

                #  SelectKBest
                st.write(" Running SelectKBest...")
                sel = SelectKBest(score_func=f_classif, k=min(10, X_processed.shape[1]))
                X_sel = sel.fit_transform(X_processed, y)
                acc = cross_val_score(clf, X_sel, y, cv=5, scoring="accuracy").mean()
                results["SelectKBest"] = acc

                #  PCA
                st.write(" Running PCA...")
                pca = PCA(n_components=min(10, X_processed.shape[1]))
                X_pca = pca.fit_transform(X_processed)
                acc_pca = cross_val_score(clf, X_pca, y, cv=5, scoring="accuracy").mean()
                results["PCA"] = acc_pca

                #  Genetic Algorithm Feature Selection
                st.write(" Running GA feature selection...")
                clf = LogisticRegression(max_iter=500)
                ga = GAFeatureSelector(
                    estimator=clf,
                    n_gen=5,
                    pop_size=10,
                )
                ga.fit(X_processed.values, y.values)

                results["Genetic Algorithm"] = ga.best_score_

                selected_mask = ga.best_mask_
                feature_names = X_processed.columns
                
                ga_features_df = pd.DataFrame({
                    "Feature": feature_names,
                    "Selected by GA": ["✅" if m == 1 else "❌" for m in selected_mask]
                })
                
                chosen = ga_features_df[ga_features_df["Selected by GA"] == "✅"]
                st.subheader("GA الميزات التي تم اختيارها بواسطة  ")
                st.dataframe(chosen.reset_index(drop=True))
                
                
                st.subheader(" مقارنة النتائج:")
                st.write(pd.DataFrame.from_dict(results, orient="index", columns=["Accuracy"]))

                joblib.dump(results, "results.joblib")
                st.info("results.joblib تم حفظ النتائج في ")

            except Exception as e:
                st.error(f" حدث خطأ أثناء التحليل: {e}")

else:
    st.info(" لبدء التحليلCSVارفع ملف ")