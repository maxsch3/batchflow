from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class IdentityEncoder(BaseEstimator, TransformerMixin):

    def __init__(self):
        super().__init__()

    def fit_transform(self, X, y=None, **fit_params):
        return self

    @staticmethod
    def transform(X):
        return X.values
