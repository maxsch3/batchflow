import numpy as np
from .batch_transformer import BatchTransformer


class FeatureDropout(BatchTransformer):

    def __init__(self, row_prob, cols, drop_values, col_probs=None):
        super().__init__()
        if type(row_prob) not in [float]:
            raise ValueError('Error: row_prob must be a scalar float')
        self._row_prob = row_prob

        # checking cols and drop_values
        self._cols = cols
        self._drop_values = drop_values
        if type(self._cols) is list:
            if not all([type(s) == str for s in self._cols]):
                raise ValueError('Error: parameter cols can only contain strings')
            if type(drop_values) is list:
                if len(drop_values) != len(self._cols):
                    raise ValueError('Error: when parameter drop_values is a list, it has to be the same length '
                                     'as cols')
            else:
                # if drop values is a single value it will be used for all columns
                self._drop_values = [self._drop_values]*len(self._cols)
            self._cols = self._cols
        elif type(self._cols) == str:
            if not np.isscalar(drop_values):
                raise ValueError('Error: when cols is string, parameter drop_values must be a scalar')
            self._cols = [self._cols]
            self._drop_values = [self._drop_values]
        else:
            raise ValueError('Error: parameter cols must be a single column name a list of column names')

        # checking col_probs
        if col_probs is not None:
            if type(col_probs) is list:
                if type(cols) is not list:
                    raise ValueError('Error: when cols is a string, col_probs must be None')
                if len(col_probs) != len(cols):
                    raise ValueError('Error: col_probs length must match length of cols list')
            else:
                raise ValueError('Error: parameter col_probs can only be a list of floats or None')
        self._col_probs = col_probs

    def transform(self, batch):
        batch = batch.apply(self._transform_row, axis=1)
        return batch

    def _transform_row(self, row):
        if np.random.binomial(1, self._row_prob):
            idx = np.random.choice(np.arange(len(self._cols)), p=self._col_probs)
            row[self._cols[idx]] = self._drop_values[idx]
        return row
