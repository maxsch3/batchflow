import pytest
import pandas as pd
import numpy as np
from keras_batchflow.batch_generator import BatchGenerator
from keras_batchflow.batch_transformer import BatchTransformer
from sklearn.preprocessing import LabelEncoder, LabelBinarizer, OneHotEncoder


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
        assert batch[1].shape == (3, )
        batch = bg[2]
        assert type(batch) == tuple
        assert type(batch[0]) == np.ndarray
        assert type(batch[1]) == np.ndarray
        assert batch[0].shape == (2, 3)
        assert batch[1].shape == (2, )
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
            batch_size=self.df.shape[0] +10,
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
        TODO: need to add actual transformer here
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

        bt1 = BatchTransformer()
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

if __name__ == '__main__':
    pytest.main([__file__])
