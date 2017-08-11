
import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

def fetch_pid(url="https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"):
    names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
    dataframe = pd.read_csv(url, names=names)
    array = dataframe.values
    X = array[:,0:8]
    y = array[:,8]
    return X, y


def pid_lr(seed=42, scoring='f1'):
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    model = LogisticRegression()
    scoring = 'f1'
    results = model_selection.cross_val_score(model, X, y, cv=kfold, scoring=scoring)
    mean_f1 = results.mean()
    std_f1 = results.std()
    return mean_f1, std_f1

X, y = fetch_pid()
mean_f1, std_f1 = pid_lr()

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
