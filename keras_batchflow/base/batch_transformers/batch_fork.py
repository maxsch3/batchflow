from .batch_transformer import BatchTransformer
import numpy as np
import pandas as pd


class BatchFork(BatchTransformer):

    def __init__(self, levels=('x', 'y'), **kwargs):
        super(BatchFork, self).__init__(**kwargs)
        self._levels = self._verify_levels(levels)

    def _verify_levels(self, levels):
        if type(levels) not in [list, tuple, np.ndarray]:
            raise ValueError('Error. Parameters levels must be a list, a tuple or a numpy array')
        return tuple(levels)

    def transform(self, batch):
        batch = pd.concat([batch]*len(self._levels), axis=1, keys=self._levels)
        return batch

    def inverse_transform(self, batch):
        return batch
