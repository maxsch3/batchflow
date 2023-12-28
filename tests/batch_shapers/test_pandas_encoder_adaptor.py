import pytest
import numpy as np
import pandas as pd

from keras_batchflow.base.batch_shapers.pandas_encoder_adaptor import PandasEncoderAdaptor


class TestPandasEncoderAdaptor:

    def test_init(self):
        nea = PandasEncoderAdaptor()

    def test_transform(self):
        data = pd.Series([1, 2, 4, 5])
        pea = PandasEncoderAdaptor()
        tr = pea.transform(data)
        assert isinstance(tr, pd.Series)
        assert tr.ndim == 1
        assert tr.shape == data.shape

    def test_transform_integer_array(self):
        """
        This tests that pandas specific IntegerArray is converted into numpy format
        :return:
        """
        data = pd.Series([1, 2, 4, 5], dtype="Int64")
        pea = PandasEncoderAdaptor()
        tr = pea.transform(data)
        assert isinstance(tr, pd.Series)
        assert tr.dtype == "Int64"

    def test_inverse_transform(self):
        data = pd.Series([1, 2, 3])
        pea = PandasEncoderAdaptor()
        tr = pea.inverse_transform(data)
        assert isinstance(tr, pd.Series)
        assert np.issubdtype(tr.dtype, np.int64)
        tr = pea.inverse_transform(data, dtype=np.float32)
        assert isinstance(tr, pd.Series)
        assert np.issubdtype(tr.dtype, np.float32)
        tr = pea.inverse_transform(data, dtype="float32")
        assert np.issubdtype(tr.dtype, np.float32)
        tr = pea.inverse_transform(data, dtype="Int64")
        assert tr.dtype == "Int64"

    def test_inverse_transform_error_handling(self):
        # Check wrong types
        pea = PandasEncoderAdaptor()
        data = np.array([1, 2, 3])
        with pytest.raises(TypeError):
            _ = pea.inverse_transform(data)

