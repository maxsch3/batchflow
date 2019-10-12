import numpy as np
import pandas as pd
from keras.utils import Sequence

class BatchGenerator(Sequence):

    """
    root class for batch generators.
    """

    def __init__(self, data: pd.DataFrame, transform_stack, batch_size=32, shuffle=True, train_mode=True):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.train_mode = train_mode
        self.indices = np.arange(self.data.shape[0])
        if type(transform_stack) is not list:
            raise ValueError('Error: transform stack must be a list')
        self.transform_stack = transform_stack
        self.on_epoch_end()

    def __len__(self):
        """
        Helps to determine number of batches in one epoch
        :return:
        n - number of batches
        """
        return int(np.ceil(self.data.shape[0] / self.batch_size))

    def __getitem__(self, index):
        """
        Calculates and returns a batch with index=index
        :param index: int between 0 and length of the epoch returned by len(self)
        :return:
        tuple (X, y) if train_mode = True, or just X otherwise
        Structures of X and y are defined by instance of BatchShaper class used in constructor
        """
        batch = self.__select_batch(index)
        return self.__transform_batch(batch)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __select_batch(self, index):
        start_pos = index * self.batch_size
        if start_pos >= len(self.indices):
            raise RuntimeError('Error: index out of bounds when selecting next batch in {}'.format(type(self).__name__))
        batch_idx = self.indices[start_pos:min(start_pos + self.batch_size, len(self.indices))]
        return self.data.iloc[batch_idx].copy()

    def __transform_batch(self, batch):
        xy = batch
        for t in self.transform_stack:
            xy = t.transform(xy, return_y=self.train_mode)
        return xy
