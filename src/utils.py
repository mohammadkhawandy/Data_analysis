# src/utils.py
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from difflib import get_close_matches

def load_csv(path_or_buffer, target_col=None, sep=None):
    
    try:
        if isinstance(path_or_buffer, str):
            # detect simple sep by peeking first 2048 bytes
            if sep is None:
                with open(path_or_buffer, "rb") as f:
                    sample = f.read(2048).decode(errors="ignore")
                sep = ";" if ";" in sample and sample.count(";") > sample.count(",") else ","
            df = pd.read_csv(path_or_buffer, encoding="utf-8-sig", sep=sep)
        else:
            # UploadedFile or buffer-like
            if sep is None:
                # try semicolon first then comma
                try:
                    df = pd.read_csv(path_or_buffer, encoding="utf-8-sig", sep=";")
                except Exception:
                    path_or_buffer.seek(0)
                    df = pd.read_csv(path_or_buffer, encoding="utf-8-sig", sep=",")
            else:
                df = pd.read_csv(path_or_buffer, encoding="utf-8-sig", sep=sep)
    except Exception as e:
        raise RuntimeError(f"Error reading CSV: {e}")

    # normalize column names (strip)
    df.columns = [str(c).strip() for c in df.columns]

    if target_col is None:
        return df

    # try exact match, case-insensitive, fuzzy
    if target_col in df.columns:
        X = df.drop(columns=[target_col])
        y = df[target_col]
        return X, y

    lower_map = {c.lower(): c for c in df.columns}
    if target_col.lower() in lower_map:
        real = lower_map[target_col.lower()]
        X = df.drop(columns=[real])
        y = df[real]
        return X, y

    matches = get_close_matches(target_col, df.columns.tolist(), n=3, cutoff=0.5)
    if matches:
        real = matches[0]
        X = df.drop(columns=[real])
        y = df[real]
        print(f"Warning: using close match '{real}' for target '{target_col}'")
        return X, y

    raise KeyError(f"Target column '{target_col}' not found. Available columns: {df.columns.tolist()}")


def preprocess(X_train, X_test):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    return X_train_s, X_test_s, scaler

def evaluate_model(clf, X, y, cv=5):
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    accs = []
    f1s = []
    precs = []
    recs = []
    for train_idx, test_idx in skf.split(X, y):
        Xtr, Xte = X[train_idx], X[test_idx]
        ytr, yte = y[train_idx], y[test_idx]
        clf.fit(Xtr, ytr)
        ypred = clf.predict(Xte)
        accs.append(accuracy_score(yte, ypred))
        f1s.append(f1_score(yte, ypred, average='weighted'))
        precs.append(precision_score(yte, ypred, average='weighted', zero_division=0))
        recs.append(recall_score(yte, ypred, average='weighted'))
    return {
        "accuracy": np.mean(accs),
        "f1": np.mean(f1s),
        "precision": np.mean(precs),
        "recall": np.mean(recs)
    }