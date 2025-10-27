import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression, LassoCV
from utils import evaluate_model


def pca_pipeline(estimator, X, y, n_components=10, cv=5):
    pca = PCA(n_components=n_components)
    Xp = pca.fit_transform(X)
    return evaluate_model(estimator, Xp, y, cv=cv), pca

def selectk_pipeline(estimator, X, y, k=10, cv=5):
    sel = SelectKBest(score_func=f_classif, k=k)
    Xs = sel.fit_transform(X, y)
    return evaluate_model(estimator, Xs, y, cv=cv), sel

def lasso_pipeline(estimator, X, y, cv=5):
    
    from sklearn.linear_model import LogisticRegression
    sel = LogisticRegression(penalty='l1', solver='saga', max_iter=5000, C=1.0)
    res = evaluate_model(sel, X, y, cv=cv)
    return res, sel