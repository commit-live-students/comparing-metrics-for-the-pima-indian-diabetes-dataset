from unittest import TestCase
import numpy as np
import pandas as pd


def fetch_pid(url="./data/pima-indians-diabetes.csv"):
    names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
    dataframe = pd.read_csv(url, names=names)
    array = dataframe.values
    X = array[:, 0:8]
    y = array[:, 8]
    return ((X, y))


class TestTraining_report(TestCase):
    def test_training_report(self):
        from build import training_report
        X, y = fetch_pid()
        from sklearn.linear_model import LogisticRegression
        out = training_report(X=X, y=y, seed=42, KFold=10, classifier=LogisticRegression())
        self.assertTrue(isinstance(out, pd.DataFrame))
