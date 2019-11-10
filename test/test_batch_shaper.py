import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, LabelBinarizer, OneHotEncoder
from keras_batchflow.batch_shaper.batch_shaper import BatchShaper


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

    def test_init_with_data_sample(self):
        # TODO
        pass

    def test_none_transformer(self):
        bs = BatchShaper(x_structure=[('var1', self.lb), ('var2', None)], y_structure=('label', self.le))
        batch = bs.transform(self.df)
        assert type(batch) == tuple
        assert len(batch) == 2
        assert type(batch[0]) == list
        assert len(batch[0]) == 2
        assert np.array_equal(batch[0][1], self.df['var2'].values)

    def test_const_component_int(self):
        bs = BatchShaper(x_structure=[('var1', self.lb), (None, 0)], y_structure=('label', self.le))
        batch = bs.transform(self.df)
        assert type(batch) == tuple
        assert len(batch) == 2
        assert type(batch[0]) == list
        assert len(batch[0]) == 2
        assert np.all(batch[0][1] == 0)
        assert batch[0][1].dtype == int

    def test_const_component_float(self):
        bs = BatchShaper(x_structure=[('var1', self.lb), (None, 0.)], y_structure=('label', self.le))
        batch = bs.transform(self.df)
        assert type(batch) == tuple
        assert len(batch) == 2
        assert type(batch[0]) == list
        assert len(batch[0]) == 2
        assert np.all(batch[0][1] == 0)
        assert batch[0][1].dtype == float

    def test_const_component_str(self):
        bs = BatchShaper(x_structure=[('var1', self.lb), (None, u'a')], y_structure=('label', self.le))
        batch = bs.transform(self.df)
        assert type(batch) == tuple
        assert len(batch) == 2
        assert type(batch[0]) == list
        assert len(batch[0]) == 2
        assert np.all(batch[0][1] == 'a')
        assert batch[0][1].dtype == '<U1' # single unicode character

    def test_metadata(self):
        bs = BatchShaper(x_structure=[('var1', self.lb), (None, 0.)], y_structure=('label', self.le))
        md = bs.get_metadata(self.df)
        assert type(md) is tuple
        assert len(md) == 2
        assert type(md[0]) is list
        assert len(md[0]) == 2
        assert type(md[0][0]) == dict
        assert type(md[0][1]) == dict
        fields_in_meta = ['name', 'encoder', 'shape', 'dtype']
        assert all([all([f in m for f in fields_in_meta]) for m in md[0]])
        assert md[0][0]['name'] == 'var1'
        assert md[0][0]['encoder'] == self.lb
        assert md[0][0]['shape'] == (None, 3)
        assert md[0][0]['dtype'] == np.int64
        assert md[0][1]['name'] == 'dummy_constant_0'
        assert md[0][1]['encoder'] is None
        assert md[0][1]['shape'] == (None, 1)
        assert md[0][1]['dtype'] == float
        assert type(md[1]) == dict
        assert all([f in md[1] for f in fields_in_meta])
        assert md[1]['name'] == 'label'
        assert md[1]['encoder'] == self.le
        assert md[1]['shape'] == (None, 1)
        assert md[1]['dtype'] == np.int64

    def test_dummy_var_naming(self):
        bs = BatchShaper(x_structure=[('var1', self.lb), (None, 0.), (None, 1.)], y_structure=('label', self.le))
        md = bs.get_metadata(self.df)
        assert type(md) is tuple
        assert len(md) == 2
        assert type(md[0]) is list
        assert len(md[0]) == 3
        assert all([type(m) == dict for m in md[0]])
        assert md[0][1]['name'] == 'dummy_constant_0'
        assert md[0][2]['name'] == 'dummy_constant_1'
        # test the counter resets with new metadata request
        md = bs.get_metadata(self.df)
        assert md[0][1]['name'] == 'dummy_constant_0'
        assert md[0][2]['name'] == 'dummy_constant_1'

    def test_shape(self):

        class A:
            @property
            def shape(self):
                return None, 11

            def transform(self, data):
                return data.values

        a = A()
        bs = BatchShaper(x_structure=[('var1', self.lb), ('var1', a)], y_structure=('label', self.le))
        # At this point, shape is not yet measured (fitted) and runtime error is expected
        with pytest.raises(RuntimeError):
            batch = bs.shape
        bs.fit_shapes(self.df)
        shapes = bs.shape
        assert type(shapes) == tuple
        assert type(shapes[0]) == list
        assert len(shapes[0]) == 2
        assert shapes[0][0] == (None, 3)    # measured
        assert shapes[0][1] == (None, 11)   # direct from transformer's shape property
        assert shapes[1] == (None, 1)       # one dimensional output

    def test_n_classes(self):

        class A:
            @property
            def n_classes(self):
                return 13

            def transform(self, data):
                return data.values

        a = A()
        bs = BatchShaper(x_structure=[('var1', self.lb), ('var1', a)], y_structure=('label', self.le))
        n_classes = bs.n_classes
        pass

    def test_inverse_transform(self):
        le2 = LabelEncoder().fit(self.df['var2'])
        bs = BatchShaper(x_structure=('var1', self.lb), y_structure=[('label', self.le), ('var2', le2)])
        batch = bs.transform(self.df)
        inverse = bs.inverse_transform(batch[1])
        assert inverse.equals(self.df[['label', 'var2']])
