import pytest
import pandas as pd
import numpy as np
from keras_batchflow.base.batch_generators import BatchGenerator
from keras_batchflow.base.batch_transformers import BatchTransformer
from sklearn.preprocessing import LabelEncoder, LabelBinarizer, OneHotEncoder

from keras_batchflow.base.batch_shapers.numpy_encoder_adaptor import NumpyEncoderAdaptor
from keras_batchflow.base.batch_shapers.pandas_encoder_adaptor import PandasEncoderAdaptor


class TestBatchGenerator:

    df = None
    le = LabelEncoder()
    lb = LabelBinarizer()
    oh = OneHotEncoder()

    def setup_method(self):
        self.df = pd.DataFrame({
            'var1': ['Class 0', 'Class 1', 'Class 0', 'Class 2', 'Class 0', 'Class 1', 'Class 0', 'Class 2'],
            'var2': ['Green', 'Yellow', 'Red', 'Brown', 'Green', 'Yellow', 'Red', 'Brown'],
            'label': ['Leaf', 'Flower', 'Leaf', 'Branch', 'Green', 'Yellow', 'Red', 'Brown']
        })
        self.le.fit(self.df['label'])
        self.oh.fit(self.df[['var1', 'var2']])
        self.lb.fit(self.df['var1'])

    def teardown_method(self):
        pass

    def test_basic(self):
        bg = BatchGenerator(
            self.df,
            x_structure=('var1', self.lb),
            y_structure=('label', self.le),
            batch_size=3,
            shuffle=True,
        )
        assert len(bg) == 3
        batch = bg[0]
        assert type(batch) == tuple
        assert type(batch[0]) == np.ndarray
        assert type(batch[1]) == np.ndarray
        assert batch[0].shape == (3, 3)
        assert batch[1].shape == (3, 1)
        batch = bg[2]
        assert type(batch) == tuple
        assert type(batch[0]) == np.ndarray
        assert type(batch[1]) == np.ndarray
        assert batch[0].shape == (2, 3)
        assert batch[1].shape == (2, 1)
        with pytest.raises(IndexError):
            batch = bg[3]

    def test_batch_sizes(self):
        # batch size equals to dataset size
        bg = BatchGenerator(
            self.df,
            x_structure=('var1', self.lb),
            y_structure=('label', self.le),
            batch_size=self.df.shape[0],
            shuffle=True,
        )
        batch = bg[0]
        assert batch[0].shape[0] == self.df.shape[0]
        # batch size bigger than dataset size
        bg = BatchGenerator(
            self.df,
            x_structure=('var1', self.lb),
            y_structure=('label', self.le),
            batch_size=self.df.shape[0] + 10,
            shuffle=True,
        )
        batch = bg[0]
        assert batch[0].shape[0] == self.df.shape[0]

    def test_shuffle(self):
        bg = BatchGenerator(
            self.df,
            x_structure=('var1', self.lb),
            y_structure=('label', self.le),
            batch_size=8,
            shuffle=True,
        )
        batch1 = bg[0]
        bg.on_epoch_end()
        bg.on_epoch_end()
        batch2 = bg[0]
        assert not np.array_equal(batch1[0], batch2[0])

    def test_no_shuffle(self):
        bg = BatchGenerator(
            self.df,
            x_structure=('var1', self.lb),
            y_structure=('label', self.le),
            batch_size=8,
            shuffle=False,
        )
        batch1 = bg[0]
        bg.on_epoch_end()
        bg.on_epoch_end()
        batch2 = bg[0]
        assert np.array_equal(batch1[0], batch2[0])

    def test_metadata(self):
        bg = BatchGenerator(
            self.df,
            x_structure=('var1', self.lb),
            y_structure=('label', self.le),
            batch_size=8,
            shuffle=False,
        )
        md = bg.metadata
        assert type(md) == tuple
        assert len(md) == 2
        assert type(md[0]) == dict
        assert type(md[1]) == dict
        # more thorough tests of the metadata format are in BatchShaper tests

    def test_batch_transformer_integration(self):
        """
        This test only checks that declaring and using BatchGenerator with batch_transform parameter
        does not cause errors.
        TODO: need to add actual encoders here
        """
        class TestTransform(BatchTransformer):
            def __init__(self, col_name):
                self.col_name = col_name
                super().__init__()

            def transform(self, batch):
                # red is the most rare label in the dataset. Here I will blindly replace all values in
                # col_name column with this value
                batch[self.col_name] = 'Red'
                return batch

            def inverse_transform(self, batch):
                return batch

        class TransparentTransform(BatchTransformer):

            def transform(self, batch):
                return batch

            def inverse_transform(self, batch):
                return batch

        bt1 = TransparentTransform()
        bt2 = TestTransform('label')
        bg = BatchGenerator(
            self.df,
            x_structure=('var1', self.lb),
            y_structure=('label', self.le),
            batch_transforms=[bt1, bt2],
            batch_size=8,
            shuffle=False,
        )
        batch = bg[0]
        assert type(batch) == tuple
        assert len(batch) == 2
        assert (self.le.inverse_transform(batch[1]) == 'Red').all()

    def test_transform(self):
        le = LabelEncoder().fit(self.df['var2'])
        bg = BatchGenerator(
            self.df,
            x_structure=[('var1', self.lb), ('var2', le)],
            y_structure=('label', self.le),
            shuffle=False,
            batch_size=3,
        )
        batch = bg.transform(self.df)
        assert type(batch) == tuple
        assert len(batch) == 2
        assert type(batch[0]) == list
        assert len(batch[0]) == 2
        assert type(batch[1]) == np.ndarray
        assert batch[1].shape == (8, 1)
        batch = bg.transform(self.df, return_y=False)
        assert isinstance(batch, tuple)
        assert len(batch) == 1
        assert type(batch[0]) == list
        assert len(batch[0]) == 2

    def test_inverse_transform(self):
        # batch size equals to dataset size
        bg = BatchGenerator(
            self.df,
            x_structure=('label', self.le),
            y_structure=('var1', self.lb),
            batch_size=self.df.shape[0],
            shuffle=False,
        )
        batch = bg[0]
        inverse = bg.inverse_transform(batch[1])
        assert type(inverse) is pd.DataFrame
        assert inverse.shape == (self.df.shape[0], 1)
        b = self.df[['var1']]
        assert inverse.equals(self.df[['var1']])

    def test_y_structure_none(self):
        bg = BatchGenerator(
            self.df,
            x_structure=('label', self.le),
            batch_size=self.df.shape[0],
            shuffle=False,
        )
        batch = bg[0]
        assert isinstance(batch, tuple)
        assert len(batch) == 1
        assert isinstance(batch[0], np.ndarray)

    def test_shapes(self):
        le = LabelEncoder().fit(self.df['var2'])
        bg = BatchGenerator(
            self.df,
            x_structure=[('var1', self.lb), ('var2', le)],
            y_structure=('label', self.le),
            shuffle=False,
        )
        sh = bg.shapes
        assert type(sh) == tuple
        assert len(sh) == 2
        assert type(sh[0]) == list
        assert len(sh[0]) == 2
        assert sh[0][0] == (3,)
        assert sh[0][1] == (1,)
        assert sh[1] == (1,)

    def test_encoder_adaptor(self):
        """
        This test only makes sure the adaptor parameter is passed correctly
        :return:
        """
        bg = BatchGenerator(
            self.df,
            x_structure=('var1', self.lb),
            y_structure=('label', self.le),
            shuffle=False,
            encoder_adaptor='numpy'
        )
        assert isinstance(bg.batch_shaper.x_structure._encoder_adaptor, NumpyEncoderAdaptor)
        bg = BatchGenerator(
            self.df,
            x_structure=('var1', self.lb),
            y_structure=('label', self.le),
            shuffle=False,
            encoder_adaptor='pandas'
        )
        assert isinstance(bg.batch_shaper.x_structure._encoder_adaptor, PandasEncoderAdaptor)

if __name__ == '__main__':
    pytest.main([__file__])
