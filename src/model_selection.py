from sklearn.model_selection import StratifiedKFold, cross_val_score

def cross_validation(estimator, X, y, cv=5, random_state=42):
    kfolds = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    scores = cross_val_score(estimator, X, y, cv=kfolds, scoring="accuracy")
    return scores