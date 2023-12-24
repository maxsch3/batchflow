import pytest
import numpy as np
import pandas as pd

from keras_batchflow.base.batch_shapers.numpy_encoder_adaptor import NumpyEncoderAdaptor


class TestNumpyEncoderAdaptor:

    def test_init(self):
        nea = NumpyEncoderAdaptor()

    def test_transform(self):
        data = pd.Series([1, 2, 4, 5])
        nea = NumpyEncoderAdaptor()
        tr = nea.transform(data)
        assert isinstance(tr, np.ndarray)
        assert tr.ndim == 1
        assert tr.shape == data.shape

    def test_transform_integer_array(self):
        """
        This tests that pandas specific IntegerArray is converted into numpy format.
        IntergerArray of type "Int64" in pandas is a very handy integer data type which supports Nones and does
        not require conversion to float data type
        :return:
        """
        data = pd.Series([1, 2, 4, 5], dtype="Int64")
        nea = NumpyEncoderAdaptor()
        tr = nea.transform(data)
        assert isinstance(tr, np.ndarray)
        assert np.issubdtype(tr.dtype, object)

    def test_transform_datetime(self):
        """
        This tests that pandas datetime types are converting correctly
        :return:
        """
        data = pd.Series(['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04'], dtype='datetime64[ns]')
        nea = NumpyEncoderAdaptor()
        tr = nea.transform(data)
        assert isinstance(tr, np.ndarray)
        assert np.issubdtype(tr.dtype, np.datetime64)

    def test_inverse_transform(self):
        data = np.array([1, 2, 3])
        nea = NumpyEncoderAdaptor()
        tr = nea.inverse_transform(data)
        assert isinstance(tr, pd.Series)
        assert np.issubdtype(tr.dtype, np.int64)
        tr = nea.inverse_transform(data, dtype=np.float32)
        assert isinstance(tr, pd.Series)
        assert np.issubdtype(tr.dtype, np.float32)
        tr = nea.inverse_transform(data, dtype="float32")
        assert np.issubdtype(tr.dtype, np.float32)
        tr = nea.inverse_transform(data, dtype="Int64")
        assert tr.dtype == "Int64"

    def test_inverse_transform_error_handling(self):
        # check 2d data squeesable to 1D
        data = np.array([[1, 2, 3]])
        nea = NumpyEncoderAdaptor()
        tr = nea.inverse_transform(data)
        assert isinstance(tr, pd.Series)
        assert len(tr) == data.shape[1]
        # try non-squeesable 2d data
        data = np.array([[1, 2, 3], [4, 5, 6]])
        with pytest.raises(ValueError):
            _ = nea.inverse_transform(data)
        # Check wrong types
        data = pd.Series([1, 2, 3])
        with pytest.raises(TypeError):
            _ = nea.inverse_transform(data)

