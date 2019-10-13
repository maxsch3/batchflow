import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, LabelBinarizer, OneHotEncoder
from batch_shaper.batch_shaper import BatchShaper


class TestBatchShaper:

    df = None
    le = LabelEncoder()
    lb = LabelBinarizer()
    oh = OneHotEncoder()

    def setup_method(self):
        self.df = pd.DataFrame({
            'var1': ['Class 0', 'Class 1', 'Class 0', 'Class 2'],
            'var2': ['Green', 'Yellow', 'Red', 'Brown'],
            'label': ['Leaf', 'Flower', 'Leaf', 'Branch']
        })
        self.le.fit(self.df['label'])
        self.oh.fit(self.df[['var1', 'var2']])
        self.lb.fit(self.df['var1'])

    def teardown_method(self):
        pass

    def test_basic(self):
        bs = BatchShaper(x_structure=('var1', self.lb), y_structure=('label', self.le))
        batch = bs.transform(self.df)
        assert type(batch) == tuple
        assert len(batch) == 2
        assert type(batch[0]) == np.ndarray
        assert type(batch[1]) == np.ndarray
        assert batch[0].shape == (4, 3)
        assert batch[1].shape == (4,)

    def test_no_return_y(self):
        bs = BatchShaper(x_structure=('var1', self.lb), y_structure=('label', self.le))
        kwargs = {'return_y': False}
        batch = bs.transform(self.df, **kwargs)
        assert type(batch) == np.ndarray
        assert batch.shape == (4, 3)

    def test_2d_transformer(self):
        """
        this test checks if a BatchShaper will throw a ValueError exception when a 2D transformer is used,
        e.g. OneHotEncoder. It requires 2D input, while BatchShaper only works on per-column basis, i.e.
        provides only 1D data.
        :return:
        """
        bs = BatchShaper(x_structure=('var1', self.oh), y_structure=('label', self.le))
        with pytest.raises(ValueError):
            batch = bs.transform(self.df)

    def test_many_x(self):
        lb2 = LabelBinarizer().fit(self.df['var2'])
        bs = BatchShaper(x_structure=[('var1', self.lb), ('var2', lb2)], y_structure=('label', self.le))
        batch = bs.transform(self.df)
        assert type(batch) == tuple
        assert len(batch) == 2
        assert type(batch[0]) == list
        assert type(batch[1]) == np.ndarray
        assert len(batch[0]) == 2
        assert type(batch[0][0]) == np.ndarray
        assert type(batch[0][1]) == np.ndarray
        assert batch[0][0].shape == (4, 3)
        assert batch[0][1].shape == (4, 4)
        assert batch[1].shape == (4,)

    def test_many_y(self):
        lb2 = LabelBinarizer().fit(self.df['var2'])
        bs = BatchShaper(x_structure=('var1', self.lb), y_structure=[('label', self.le), ('var2', lb2)])
        batch = bs.transform(self.df)
        assert type(batch) == tuple
        assert len(batch) == 2
        assert type(batch[0]) == np.ndarray
        assert type(batch[1]) == list
        assert len(batch[1]) == 2
        assert type(batch[1][0]) == np.ndarray
        assert type(batch[1][1]) == np.ndarray
        assert batch[1][0].shape == (4,)
        assert batch[1][1].shape == (4, 4)
        assert batch[0].shape == (4, 3)

    def test_wrong_format(self):
        lb2 = LabelBinarizer().fit(self.df['var2'])
        # this must throw ValueError - leafs of a structure must be tuples of
        # format ('column name', transformer_instance)
        bs = BatchShaper(x_structure=('var1', self.lb), y_structure=('label', self.le, 1))
        # this must throw ValueError - leafs of a structure must be tuples of
        # format ('column name', transformer_instance)
        bs = BatchShaper(x_structure=('var1', self.lb), y_structure=('label', 1))
        with pytest.raises(ValueError):
            batch = bs.transform(self.df)
        # this must also throw ValueError - structure must be a tuple (X, y) to conform Keras requirements
        bs = BatchShaper(x_structure=[('var1', self.lb)], y_structure=('label', self.le, 1))
        with pytest.raises(ValueError):
            batch = bs.transform(self.df)

    def test_missing_field(self):
        bs = BatchShaper(x_structure=('missing_name', self.lb), y_structure=('label', self.le, 1))
        with pytest.raises(KeyError):
            batch = bs.transform(self.df)

    def test_shape(self):
        lb2 = LabelBinarizer().fit(self.df['var2'])
        bs = BatchShaper(x_structure=[('var1', self.lb), ('var2', lb2)], y_structure=('label', self.le))
        # At this point, shape is not yet measured (fitted) and runtime error is expected
        with pytest.raises(RuntimeError):
            batch = bs.shape
        bs.fit_shapes(self.df)
        shape = bs.shape
        assert type(shape) == tuple
        assert len(shape) == 2
        assert type(shape[0]) == list
        assert len(shape[0]) == 2
        assert shape[0][0] == (None, 3)
        assert shape[0][1] == (None, 4)
        assert shape[1] == (None, )
