from .base_encoder import BaseEncoder
import numpy as np


class TripletPKBatchLabeler(BaseEncoder):

    def __init__(self):
        super().__init__()

    def fit_transform(self, X, y=None, **fit_params):
        return self

    @staticmethod
    def transform(X):
        lookup, indexed_data = np.unique(X.values, return_inverse=True)
        return indexed_data
