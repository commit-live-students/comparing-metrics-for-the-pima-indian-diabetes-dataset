import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression

seed = 7
KFold = model_selection.KFold(n_splits=10, random_state=seed)
classifier = LogisticRegression()

def training_report(X,y, seed, classifier, KFold):
    KFold = model_selection.KFold(n_splits=10, random_state=seed)

    try:
        if seed == int(seed):
            scoring = ['f1','accuracy','neg_log_loss','roc_auc']
            d = {}
            for each_score in scoring:
                results = model_selection.cross_val_score(classifier, X, y, cv=KFold, scoring=each_score)
                value = results.mean(), results.std()
                d[each_score] = value
                df = pd.DataFrame(d)
            return df
    except ValueError:
        print 'Exception occured'
# training_report(X,y, seed, classifier, kfold)
