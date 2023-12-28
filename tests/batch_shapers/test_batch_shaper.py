import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, LabelBinarizer, OneHotEncoder
from keras_batchflow.base.batch_shapers.batch_shaper import BatchShaper
from keras_batchflow.base.batch_shapers.var_shaper import VarShaper
from keras_batchflow.base.batch_transformers import BatchFork
from keras_batchflow.base.batch_shapers.numpy_encoder_adaptor import NumpyEncoderAdaptor
from keras_batchflow.base.batch_shapers.pandas_encoder_adaptor import PandasEncoderAdaptor


class TestBatchShaper:

    @pytest.fixture
    def data(self):
        return pd.DataFrame({
            'var1': ['Class 0', 'Class 1', 'Class 0', 'Class 2'],
            'var2': ['Green', 'Yellow', 'Red', 'Brown'],
            'label': ['Leaf', 'Flower', 'Leaf', 'Branch']
        })

    @pytest.fixture
    def label_encoder(self, data):
        encoder = LabelEncoder()
        encoder.fit(data['label'])
        return encoder

    @pytest.fixture
    def one_hot_encoder(self, data):
        encoder = OneHotEncoder()
        encoder.fit(data[['var1', 'var2']])
        return encoder

    @pytest.fixture
    def label_binarizer(self, data):
        encoder = LabelBinarizer()
        encoder.fit(data['var1'])
        return encoder

    def test_basic(self, data, label_binarizer, label_encoder):
        bs = BatchShaper(x_structure=('var1', label_binarizer), y_structure=('label', label_encoder), data_sample=data)
        batch = bs.transform(data)
        assert type(batch) == tuple
        assert len(batch) == 2
        assert type(batch[0]) == np.ndarray
        assert type(batch[1]) == np.ndarray
        assert batch[0].shape == (4, 3)
        assert batch[1].shape == (4, 1)

    def test_no_return_y(self, data, label_binarizer, label_encoder):
        bs = BatchShaper(x_structure=('var1', label_binarizer), y_structure=('label', label_encoder), data_sample=data)
        kwargs = {'return_y': False}
        batch = bs.transform(data, **kwargs)
        # the batch must have shape (ndarray,)
        assert isinstance(batch, tuple)
        assert len(batch) == 1
        assert type(batch[0]) == np.ndarray
        assert batch[0].shape == (4, 3)

    def test_2d_transformer(self, data, one_hot_encoder, label_encoder):
        """
        this test checks if a BatchShaper will throw a ValueError exception when a 2D encoders is used,
        e.g. OneHotEncoder. It requires 2D input, while BatchShaper only works on per-column basis, i.e.
        provides only 1D data.
        :return:
        """
        with pytest.raises(ValueError):
            bs = BatchShaper(x_structure=('var1', one_hot_encoder), y_structure=('label', label_encoder), data_sample=data)

    def test_many_x(self, data, label_binarizer, label_encoder):
        lb2 = LabelBinarizer().fit(data['var2'])
        bs = BatchShaper(x_structure=[('var1', label_binarizer), ('var2', lb2)],
                         y_structure=('label', label_encoder),
                         data_sample=data)
        batch = bs.transform(data)
        assert type(batch) == tuple
        assert len(batch) == 2
        assert type(batch[0]) == list
        assert type(batch[1]) == np.ndarray
        assert len(batch[0]) == 2
        assert type(batch[0][0]) == np.ndarray
        assert type(batch[0][1]) == np.ndarray
        assert batch[0][0].shape == (4, 3)
        assert batch[0][1].shape == (4, 4)
        assert batch[1].shape == (4, 1)

    def test_many_y(self, data, label_binarizer, label_encoder):
        lb2 = LabelBinarizer().fit(data['var2'])
        bs = BatchShaper(x_structure=('var1', label_binarizer),
                         y_structure=[('label', label_encoder), ('var2', lb2)],
                         data_sample=data)
        batch = bs.transform(data)
        assert type(batch) == tuple
        assert len(batch) == 2
        assert type(batch[0]) == np.ndarray
        assert type(batch[1]) == list
        assert len(batch[1]) == 2
        assert type(batch[1][0]) == np.ndarray
        assert type(batch[1][1]) == np.ndarray
        assert batch[1][0].shape == (4, 1)
        assert batch[1][1].shape == (4, 4)
        assert batch[0].shape == (4, 3)

    def test_predict_batch(self, data, label_binarizer, label_encoder):
        """
        This tests the batch used for predicting, which is missing y. This test reproduces issue 133
        (https://github.com/maxsch3/keras-batchflow/issues/133) where such batches were lists and tensorflow
        did not like it.
        https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit the documentation outlines unpacking rules
        for generators: the batch must be a tuple (x, y, sample_weight) where y and sample_weight are optional
        so for predict, the generator must return a tuple (x,), where x is a list of inputs
        """
        lb2 = LabelBinarizer().fit(data['var2'])
        batch_shaper = BatchShaper(x_structure=[('var1', label_binarizer), ('var2', lb2)], data_sample=data)
        batch = batch_shaper.transform(data)
        assert isinstance(batch, tuple)
        assert len(batch) == 1
        assert isinstance(batch[0], list)
        assert len(batch[0]) == 2



    def test_wrong_format(self, data, label_binarizer, label_encoder):
        lb2 = LabelBinarizer().fit(data['var2'])
        # this must throw ValueError - leafs of a structure must be tuples of
        # format ('column name', transformer_instance)
        with pytest.raises(ValueError):
            bs = BatchShaper(x_structure=('var1', label_binarizer), y_structure=('label', label_encoder, 1), data_sample=data)
        # this must throw ValueError - leafs of a structure must be tuples of
        # format ('column name', transformer_instance)
        with pytest.raises(ValueError):
            bs = BatchShaper(x_structure=('var1', label_binarizer), y_structure=('label', 1), data_sample=data)
        # this must also throw ValueError - structure must be a tuple (X, y) to conform Keras requirements
        with pytest.raises(ValueError):
            bs = BatchShaper(x_structure=[('var1', label_binarizer)], y_structure=('label', label_encoder, 1), data_sample=data)

    def test_missing_field(self, data, label_binarizer, label_encoder):
        with pytest.raises(KeyError):
            bs = BatchShaper(x_structure=('missing_name', label_binarizer),
                             y_structure=('label', label_encoder, 1),
                             data_sample=data)
            batch = bs.transform(data)

    def test_init_with_data_sample(self):
        # TODO
        pass

    def test_none_transformer(self, data, label_binarizer, label_encoder):
        bs = BatchShaper(x_structure=[('var1', label_binarizer), ('var2', None)],
                         y_structure=('label', label_encoder),
                         data_sample=data)
        batch = bs.transform(data)
        assert type(batch) == tuple
        assert len(batch) == 2
        assert type(batch[0]) == list
        assert len(batch[0]) == 2
        assert np.array_equal(batch[0][1], np.expand_dims(data['var2'].values, axis=-1))

    def test_const_component_int(self, data, label_binarizer, label_encoder):
        bs = BatchShaper(x_structure=[('var1', label_binarizer), (None, 0)],
                         y_structure=('label', label_encoder),
                         data_sample=data)
        batch = bs.transform(data)
        assert type(batch) == tuple
        assert len(batch) == 2
        assert type(batch[0]) == list
        assert len(batch[0]) == 2
        assert np.all(batch[0][1] == 0)
        assert batch[0][1].dtype == int

    def test_const_component_float(self, data, label_binarizer, label_encoder):
        bs = BatchShaper(x_structure=[('var1', label_binarizer), (None, 0.)],
                         y_structure=('label', label_encoder),
                         data_sample=data)
        batch = bs.transform(data)
        assert type(batch) == tuple
        assert len(batch) == 2
        assert type(batch[0]) == list
        assert len(batch[0]) == 2
        assert np.all(batch[0][1] == 0)
        assert batch[0][1].dtype == float

    def test_const_component_str(self, data, label_binarizer, label_encoder):
        bs = BatchShaper(x_structure=[('var1', label_binarizer), (None, u'a')],
                         y_structure=('label', label_encoder),
                         data_sample=data)
        batch = bs.transform(data)
        assert type(batch) == tuple
        assert len(batch) == 2
        assert type(batch[0]) == list
        assert len(batch[0]) == 2
        assert np.all(batch[0][1] == 'a')
        assert batch[0][1].dtype == '<U1'  # single unicode character

    def test_metadata(self, data, label_binarizer, label_encoder):
        VarShaper._dummy_constant_counter = 0
        bs = BatchShaper(x_structure=[('var1', label_binarizer), (None, 0.)],
                         y_structure=('label', label_encoder),
                         data_sample=data)
        md = bs.metadata
        batch = bs.transform(data)
        assert type(md) is tuple
        assert len(md) == 2
        assert type(md[0]) is list
        assert len(md[0]) == 2
        assert type(md[0][0]) == dict
        assert type(md[0][1]) == dict
        fields_in_meta = ['name', 'encoder', 'shape', 'dtype']
        assert all([all([f in m for f in fields_in_meta]) for m in md[0]])
        assert md[0][0]['name'] == 'var1'
        assert md[0][0]['encoder'] == label_binarizer
        assert md[0][0]['shape'] == (3, )
        assert batch[0][0].ndim == 2
        assert batch[0][0].shape[1] == 3
        assert md[0][0]['dtype'] == np.int64
        assert md[0][1]['name'] == 'dummy_constant_0'
        assert md[0][1]['encoder'] is None
        assert md[0][1]['shape'] == (1, )
        assert md[0][1]['dtype'] == float
        assert batch[0][1].ndim == 2
        assert type(md[1]) == dict
        assert all([f in md[1] for f in fields_in_meta])
        assert md[1]['name'] == 'label'
        assert md[1]['encoder'] == label_encoder
        assert md[1]['shape'] == (1, )
        assert batch[1].ndim == 2
        assert md[1]['dtype'] == np.int64

    def test_dummy_var_naming(self, data, label_binarizer, label_encoder):
        VarShaper._dummy_constant_counter = 0
        bs = BatchShaper(x_structure=[('var1', label_binarizer), (None, 0.), (None, 1.)],
                         y_structure=('label', label_encoder),
                         data_sample=data)
        md = bs.metadata
        assert type(md) is tuple
        assert len(md) == 2
        assert type(md[0]) is list
        assert len(md[0]) == 3
        assert all([type(m) == dict for m in md[0]])
        assert md[0][1]['name'] == 'dummy_constant_0'
        assert md[0][2]['name'] == 'dummy_constant_1'
        # test the counter resets with new metadata request
        md = bs.metadata
        assert md[0][1]['name'] == 'dummy_constant_0'
        assert md[0][2]['name'] == 'dummy_constant_1'

    def test_shape(self, data, label_binarizer, label_encoder):

        class A:
            @property
            def shape(self):
                return 11,

            def transform(self, data):
                return data

            def inverse_transform(self, data):
                return data

        a = A()
        bs = BatchShaper(x_structure=[('var1', label_binarizer), ('var1', a)],
                         y_structure=('label', label_encoder),
                         data_sample=data)
        shapes = bs.shape
        assert type(shapes) == tuple
        assert type(shapes[0]) == list
        assert len(shapes[0]) == 2
        assert shapes[0][0] == (3,)    # measured
        assert shapes[0][1] == (11,)   # direct from encoders's shape property
        assert shapes[1] == (1,)       # one dimensional output

    def test_n_classes(self, data, label_binarizer, label_encoder):

        class A:
            @property
            def n_classes(self):
                return 13

            def transform(self, data):
                return data

            def inverse_transform(self, data):
                return data

        a = A()
        bs = BatchShaper(x_structure=[('var1', label_binarizer), ('var1', a)],
                         y_structure=('label', label_encoder), data_sample=data)
        n_classes = bs.n_classes
        pass

    def test_inverse_transform(self, data, label_binarizer, label_encoder):
        le2 = LabelEncoder().fit(data['var2'])
        bs = BatchShaper(x_structure=('var1', label_binarizer),
                         y_structure=[('label', label_encoder), ('var2', le2)],
                         data_sample=data)
        batch = bs.transform(data)
        inverse = bs.inverse_transform(batch[1])
        assert inverse.equals(data[['label', 'var2']])
        # Check inverse transform when constant field is in the structure
        bs = BatchShaper(x_structure=('var1', label_binarizer),
                         y_structure=[('label', label_encoder), ('var2', le2), (None, 0.)],
                         data_sample=data)
        batch = bs.transform(data)
        # check that the constant field was added to the y output
        assert len(batch[1]) == 3
        inverse = bs.inverse_transform(batch[1])
        # this is to make sure that constant field is not decoded
        assert inverse.shape[1] == 2
        assert inverse.equals(data[['label', 'var2']])
        # Check inverse transform when direct mapping field is in the structure
        bs = BatchShaper(x_structure=('var1', label_binarizer),
                         y_structure=[('label', label_encoder), ('var2', le2), ('var1', None)],
                         data_sample=data)
        batch = bs.transform(data)
        # check that the constant field was added to the y output
        assert len(batch[1]) == 3
        inverse = bs.inverse_transform(batch[1])
        # this is to make sure that constant field is decoded
        assert inverse.shape[1] == 3
        assert inverse.equals(data[['label', 'var2', 'var1']])

    def test_multiindex_xy(self, data, label_binarizer, label_encoder):
        """ This test ensures that multiindex functionality works as expected. This function is used
        when x and y use different input data of the same structure. This is a typical scenario in
        denoising autoencoders where

        :return:
        """
        # simulate data augmentation by changing all values in column label in X to a single value
        df1 = data.copy()
        df1['label'] = df1['label'].iloc[0]
        df = pd.concat([df1, data], keys=['x', 'y'], axis=1)
        assert df.columns.nlevels == 2
        assert 'x' in df
        assert 'y' in df
        bs = BatchShaper(x_structure=('label', label_encoder), y_structure=('label', label_encoder), data_sample=data)
        batch = bs.transform(df)
        assert type(batch) == tuple
        assert len(batch) == 2
        assert type(batch[0]) == np.ndarray
        assert batch[0].shape == (4, 1)
        assert np.all(batch[0] == batch[0][0, 0])
        assert type(batch[1]) == np.ndarray
        assert batch[1].shape == batch[0].shape
        assert not np.all(batch[1] == batch[1][0, 0])

    def test_multiindex_xy_keys_input(self, data, label_binarizer, label_encoder):
        """This is to test error handling of BatchShaper with regards to multiindex_xy_keys parameter"""
        with pytest.raises(ValueError):
            _ = BatchShaper(x_structure=('label', label_encoder), y_structure=('label', label_encoder),
                            multiindex_xy_keys='x', data_sample=data)
        with pytest.raises(ValueError):
            _ = BatchShaper(x_structure=('label', label_encoder), y_structure=('label', label_encoder),
                            multiindex_xy_keys=('x', 'y', 'z'), data_sample=data)
        with pytest.raises(ValueError):
            _ = BatchShaper(x_structure=('label', label_encoder), y_structure=('label', label_encoder),
                            multiindex_xy_keys=('x', 'x'), data_sample=data)
        _ = BatchShaper(x_structure=('label', label_encoder), multiindex_xy_keys=('x', 'y'), data_sample=data)
        _ = BatchShaper(x_structure=('label', label_encoder), multiindex_xy_keys=(0, 1), data_sample=data)
        _ = BatchShaper(x_structure=('label', label_encoder), multiindex_xy_keys=(True, False), data_sample=data)

    def test_batch_forking(self, data, label_binarizer, label_encoder):
        data_snapshot = data.copy()
        batch_fork_def = BatchFork()
        data_xy_fork = batch_fork_def.transform(data)
        # check that data is not modified
        assert data.equals(data_snapshot)
        assert data_xy_fork.columns.nlevels == 2
        bs = BatchShaper(x_structure=[('var1', label_binarizer), ('label', label_encoder)],
                         y_structure=('label', label_encoder),
                         data_sample=data)
        tr = bs.transform(data_xy_fork)
        assert np.allclose(tr[0][1], tr[1])

        # now, we will test if branches are independent and can be processed separately
        data_xy_fork.loc[:, ('x', 'label')] = 'Branch'
        tr = bs.transform(data_xy_fork)
        assert not np.allclose(tr[0][1], tr[1])
        # check that only one unique value in transformed data after the source column in x structure was filled
        # with constant value
        assert np.unique(tr[0][1]).size == 1

        # test alternative multiindex keys together with BatchFork
        batch_fork_01 = BatchFork(levels=(0, 1))
        data_01_fork = batch_fork_01.transform(data)
        assert data_01_fork.columns.nlevels == 2
        bs = BatchShaper(x_structure=[('var1', label_binarizer), ('label', label_encoder)],
                         y_structure=('label', label_encoder),
                         multiindex_xy_keys=(0, 1),
                         data_sample=data)
        tr = bs.transform(data)

    def test_encoder_adaptor(self, data, label_binarizer, label_encoder):
        """
        This test checks that encoder adaptor parameter is passed correctly to a VarShaper
        """
        bs = BatchShaper(x_structure=('label', label_encoder),
                         y_structure=('label', label_encoder),
                         data_sample=data)
        # check that default is numpy adaptor
        assert isinstance(bs.x_structure._encoder_adaptor, NumpyEncoderAdaptor)
        assert isinstance(bs.y_structure._encoder_adaptor, NumpyEncoderAdaptor)
        bs = BatchShaper(x_structure=('label', label_encoder),
                         y_structure=('label', label_encoder),
                         data_sample=data,
                         encoder_adaptor='pandas')
        # check that pandas has been correctly passed to var shapers
        assert isinstance(bs.x_structure._encoder_adaptor, PandasEncoderAdaptor)
        assert isinstance(bs.y_structure._encoder_adaptor, PandasEncoderAdaptor)
