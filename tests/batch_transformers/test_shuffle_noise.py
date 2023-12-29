import pytest
import pandas as pd
import numpy as np
from scipy.stats import binom_test, chisquare
from keras_batchflow.base.batch_transformers import ShuffleNoise


class TestFeatureDropout:

    df = None

    def setup_method(self):
        self.df = pd.DataFrame({
            'var1': ['Class 0', 'Class 1', 'Class 0', 'Class 2', 'Class 0', 'Class 1', 'Class 0', 'Class 2'],
            'var2': ['Green', 'Yellow', 'Red', 'Brown', 'Green', 'Yellow', 'Red', 'Brown'],
            'label': ['Leaf', 'Flower', 'Leaf', 'Branch', 'Green', 'Yellow', 'Red', 'Brown']
        })

    def teardown_method(self):
        pass

    def test_basic(self):
        # below are all normal definitions of the transformer. they all must be successful
        sn = ShuffleNoise([.0, 1.], 'var1')
        trf = sn.transform(self.df.copy())
        assert isinstance(trf, type(self.df))
        assert not np.equal(trf.values, self.df.values).all()

    def test_multiple_columns(self):
        # there was issue #86 then multiple column setup didn't work
        sn = ShuffleNoise([.0, 1.], ['var1', 'var2', 'label'])
        trf = sn.transform(self.df.copy())
        assert isinstance(trf, type(self.df))
        assert not np.equal(trf.values, self.df.values).all()

    def test_transform_inplace(self):
        # below are all normal definitions of the transformer. they all must be successful
        sn = ShuffleNoise([.0, 1.], 'var1')
        batch = self.df.copy()
        trf = sn.transform(batch)
        assert isinstance(trf, type(batch))
        # by default transform is in place. Because it changes the source batch the transformed version will still
        # match the source
        assert np.equal(trf.values, batch.values).all()

    def test_pandas_dtype_persists(self):
        """
        This tests if the transformer keeps pandas-specific data types such as Int64. This is related to issue #114
        https://github.com/maxsch3/keras-batchflow/issues/114
        :return:
        """
        data = pd.DataFrame({'var1': np.random.randint(low=0, high=10, size=100)}).astype('Int64')
        data.iloc[0, 0] = None
        augmented_data = ShuffleNoise([.0, 1.], 'var1').transform(data.copy())
        assert all(dt.name == 'Int64' for dt in augmented_data.dtypes)
