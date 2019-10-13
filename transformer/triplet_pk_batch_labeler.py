from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class TripletPKBatchLabeler(BaseEstimator, TransformerMixin):

    def __init__(self):
        super().__init__()

    def fit_transform(self, X, y=None, **fit_params):
        return self

    @staticmethod
    def transform(X):
        lookup, indexed_data = np.unique(X.values, return_inverse=True)
        return indexed_data
