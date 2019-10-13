import numpy as np
import pandas as pd
from batch_shaper.batch_shaper import BatchShaper
from batch_transformer.batch_transformer import BatchTransformer
from keras.utils import Sequence


class BatchGenerator(Sequence):

    """
    root class for batch generators.
    """

    def __init__(self, data: pd.DataFrame, x_structure, y_structure=None,
                 batch_transforms=None, batch_size=32, shuffle=True, train_mode=True):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.train_mode = train_mode
        self.batch_shaper = BatchShaper(x_structure, y_structure,
                                        data_sample=data.iloc[:min(data.shape[0], 10)])
        self.__check_batch_transformers(batch_transforms)
        self.batch_transforms = batch_transforms
        self.indices = np.arange(self.data.shape[0])
        self.on_epoch_end()

    def __check_batch_transformers(self, batch_transformers):
        if batch_transformers is None:
            pass
        elif type(batch_transformers) == list:
            if not all([issubclass(type(t), BatchTransformer)for t in batch_transformers]):
                raise ValueError('Error: all batch transformers must be derived from BatchTransformer class')
        elif not issubclass(type(batch_transformers), BatchTransformer):
            raise ValueError('Error: batch transformer provided is not a child of BatchTransformer class')
        else:
            raise ValueError('Error: transform stack must be a list or a single batch transform object')

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
        batch = self._select_batch(index)
        return self._transform_batch(batch)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    @property
    def shape(self):
        return self.batch_shaper.shape

    def _select_batch(self, index):
        start_pos = index * self.batch_size
        if start_pos >= len(self.indices):
            raise IndexError('Error: index out of bounds when selecting next batch in {}'.format(type(self).__name__))
        batch_idx = self.indices[start_pos:min(start_pos + self.batch_size, len(self.indices))]
        return self.data.iloc[batch_idx].copy()

    def _transform_batch(self, batch):
        if self.batch_transforms is not None:
            for t in self.batch_transforms:
                batch = t.transform(batch, return_y=self.train_mode)
        return self.batch_shaper.transform(batch, return_y=self.train_mode)
