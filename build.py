
import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression

def training_report(X, y, seed, KFold, classifier):
    if type(seed) != int or type(KFold) != int:
        raise TypeError
    kfold = model_selection.KFold(n_splits=KFold, random_state=seed)
    model = classifier
    scoring = 'f1'
    results = model_selection.cross_val_score(model, X, y, cv=KFold, scoring=scoring)
    df = pd.DataFrame()
    mean_f1 = results.mean()
    std_f1 = results.std()
    df['mean_f1'] = mean_f1
    df['std_f1'] = std_f1
    df['accuracy'] = results
    return df
