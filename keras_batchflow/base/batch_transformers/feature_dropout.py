import numpy as np
import pandas as pd
from .base_random_cell import BaseRandomCellTransform


class FeatureDropout(BaseRandomCellTransform):

    def __init__(self, n_probs, cols, col_probs=None, drop_values=None):
        super(FeatureDropout, self).__init__(n_probs, cols, col_probs)
        self._drop_values = self._validate_drop_values(drop_values)

    def _validate_drop_values(self, drop_values):
        if (type(drop_values) is str) or not hasattr(drop_values, '__iter__'):
            drop_values = [drop_values]
        elif type(drop_values) not in [tuple, list, np.ndarray]:
            raise ValueError('Error: parameter cols must be a single value or list, tuple, numpy array of values')
        dv = np.array(drop_values)
        if dv.ndim > 1:
            raise ValueError('Error: drop_values must be a vector of one dimension or a scalar value')
        if (len(dv) == 1) and (len(self._cols) > 1):
            dv = np.repeat(dv, len(self._cols))
        if len(dv) != len(self._cols):
            raise ValueError('Error: drop_values and cols parameters must have same dimensions')
        return dv

    def _make_augmented_version(self, batch):
        return pd.DataFrame(np.tile(self._drop_values, (batch.shape[0], 1)), columns=self._cols, index=batch.index)
