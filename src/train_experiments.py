# src/train_experiments.py

import numpy as np
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from utils import load_csv, preprocess, evaluate_model
from ga_selector import GAFeatureSelector
from baseline import pca_pipeline, selectk_pipeline, lasso_pipeline

def run_all(dataset_path, target_col):
    df = pd.read_csv(dataset_path, delimiter=";")

    y = df[target_col]
    X = df.drop(columns=[target_col])

    cat_cols = X.select_dtypes(include=["object"]).columns
    num_cols = X.select_dtypes(exclude=["object"]).columns

    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    print(f" شكل البيانات بعد الترميز: {X.shape}")
    print(f" عدد الميزات بعد الترميز: {len(X.columns)}")

    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

    # ====================
    # Baseline Methods
    # ====================
    print(" Running SelectKBest (k=10)...")
    res_selectk, _ = selectk_pipeline(clf, X.values, y, k=min(10, X.shape[1]), cv=5)
    print("SelectKBest:", res_selectk)

    print(" Running PCA (n=10)...")
    res_pca, _ = pca_pipeline(clf, X.values, y, n_components=min(10, X.shape[1]), cv=5)
    print("PCA:", res_pca)


    # ====================
    # Genetic Algorithm
    # ====================
    print(" Running GA feature selection...")
    ga = GAFeatureSelector(estimator=clf, n_gen=20, pop_size=20, cx_prob=0.8, mut_prob=0.03, random_state=42)
    ga.fit(X.values, y, cv=3, verbose=True)

    mask = ga.best_mask_
    selected_features = list(X.columns[np.where(mask == 1)[0]])
    print(" GA selected features:", selected_features)

    X_ga = ga.transform(X.values)
    res_ga = evaluate_model(clf, X_ga, y, cv=5)
    print("GA results:", res_ga)

    joblib.dump({
        "selectk": res_selectk,
        "pca": res_pca,
        
        "ga": res_ga,
        "ga_mask": mask,
        "selected_features": selected_features
    }, "results.joblib")
    print(" Results saved to results.joblib")

if __name__ == "__main__":
    dataset_path = "data/bank-full.csv"
    target_col = "job"   
    run_all(dataset_path, target_col)