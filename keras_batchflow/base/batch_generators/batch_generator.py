import numpy as np
import pandas as pd
from ..batch_shapers.batch_shaper import BatchShaper
from ..batch_transformers.batch_transformer import BatchTransformer


class BatchGenerator:

    """ Basic batch generator. It is also a root class for all other batch generators.

    This batch generator is a python generator object that returns new data at each new iteration. It is built
    to b used in Keras's *_generator functions.

    At every new iteration, it selects small chunk of a dataset and sends it to stack of transformers specified
    at creation time. The generator makes sure that each datapoint will be selected once in one end-to-end walk

    **Parameters:**

    - **data** - a *Pandas dataframe* containing a dataset with both x and y
    - **x_structure** - *tuple* or *list of tuples* - a structure describing mapping of dataframe columns to
        pre-fitted encoders and to keras model inputs. When model has multiple inputs, keras expects
        a list of numpy arrays as model X's. Each tuple is a mapping of a dataframe column to a relevant encoder.
        It has format `('column name', encoder)`. If encoder is None, the column values will be converted to numpy
        array and passed unchanged. If `(None, value)` is used, a new constant of value = `value` will be added
        to Batch generator's output.
    - **y_structure** - (optional) *tuple* or *list of tuples* - a structure describing mapping of dataframe columns to
        pre-fitted encoders and to keras model output. When model has multiple output, keras expects
        a list of numpy arrays as model Y's. **Default: None**. Same rules and same format applies (see x_structure)
    - **batch_transforms** - (optional) *a single instance or list of BatchTransformer* - a stack of batch transformers
        that are applied to batches before splitting to columns. These are useful when variables interact during
        transform. For example, in feature dropout, when only one randomly selected feature out of multiple input
        features have to be dropped. **Default: None**
    - **batch_size** - (optional) *int* max length of generated batch. The last batch of a dataset can be smaller
        if total size of dataframe is not multiple of a batch_size. **Default: 32**
    - **shuffle** - (optional) *bool*, if true, the input dataframe is shuffled before each new epoch.
        **Default: False**
    - **train_mode** - (optional) *bool*. If true, both X and Y are returned, otherwise only X is returned
    - **encoder_adaptor** - (optional) *str* or a single instance of a class derived from
        keras_batchflow.base.batch_shapers.IEncoderAdaptor class. String values supported: 'numpy' and 'pandas'. If
        not provided, 'numpy' is used. This parameter sets format that encoders are using. Sklearn encoders are
        created for numpy arrays hence the default value is numpy. If your encoders require pandas format, use
        'pandas'. Alternatively, if your encoders need some special format, create your instance derived from
        IEncoderAdaptor class
    """

    def __init__(self, data: pd.DataFrame, x_structure, y_structure=None,
                 batch_transforms=None, batch_size=32, shuffle=True, train_mode=True, encoder_adaptor=None):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.train_mode = train_mode
        self.__check_batch_transformers(batch_transforms)
        self.batch_transforms = batch_transforms
        self.batch_shaper = BatchShaper(x_structure, y_structure,
                                        data_sample=self._apply_batch_transforms(data.iloc[:min(data.shape[0], 10)]),
                                        encoder_adaptor=encoder_adaptor)
        self.indices = np.arange(self.data.shape[0])
        self.on_epoch_end()

    def __check_batch_transformers(self, batch_transformers):
        if batch_transformers is None:
            pass
        elif type(batch_transformers) == list:
            if not all([issubclass(type(t), BatchTransformer)for t in batch_transformers]):
                raise ValueError('Error: all batch transformers must be derived from BatchTransformer class')
        elif not issubclass(type(batch_transformers), BatchTransformer):
            raise ValueError('Error: batch encoders provided is not a child of BatchTransformer class')
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

# Shape method cause error in tensorflow's method
    @property
    def shapes(self):
        return self.batch_shaper.shape

    @property
    def metadata(self):
        batch = self._select_batch(0)
        batch = self._apply_batch_transforms(batch)
        return self.batch_shaper.metadata

    def _apply_batch_transforms(self, batch):
        if self.batch_transforms is not None:
            for t in self.batch_transforms:
                batch = t.transform(batch)
        return batch

    def _select_batch(self, index):
        start_pos = index * self.batch_size
        if start_pos >= len(self.indices):
            raise IndexError('Error: index out of bounds when selecting next batch in {}'.format(type(self).__name__))
        batch_idx = self.indices[start_pos:min(start_pos + self.batch_size, len(self.indices))]
        return self.data.iloc[batch_idx].copy()

    def _transform_batch(self, batch):
        batch = self._apply_batch_transforms(batch)
        return self.batch_shaper.transform(batch, return_y=self.train_mode)

    def transform(self, data, return_y=True):
        data_bt = self._apply_batch_transforms(data)
        return self.batch_shaper.transform(data_bt, return_y=return_y)

    def inverse_transform(self, y_data):
        return self.batch_shaper.inverse_transform(y_data)
