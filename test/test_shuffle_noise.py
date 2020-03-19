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
        # by default transform is in place. Because it changes the source batch the transfrormed version will still
        # match the source
        assert np.equal(trf.values, batch.values).all()

    # def test_row_dist(self):
    #     fd = FeatureDropout(.6, 'var1', '')
    #     batch = fd.transform(self.df.sample(1000, replace=True))
    #     b = (batch['var1'] == '').sum()
    #     assert binom_test(b, 1000, 0.6) > 0.01
    #
    # def test_cols_dist(self):
    #     fd = FeatureDropout(1., ['var1', 'var2', 'label'], '', col_probs=[.5, .3, .2])
    #     batch = fd.transform(self.df.sample(1000, replace=True))
    #     b = (batch == '').sum(axis=0)
    #     c, p = chisquare(b, [500, 300, 200])
    #     assert p > 0.01
    #
    # def test_uniform_col_dist(self):
    #     fd = FeatureDropout(1., ['var1', 'var2', 'label'], '')
    #     batch = fd.transform(self.df.sample(1000, replace=True))
    #     b = (batch == '').sum(axis=0)
    #     c, p = chisquare(b, [333, 333, 333])
    #     assert p > 0.01
    #
    # def test_different_drop_values(self):
    #     fd = FeatureDropout(1., ['var1', 'var2', 'label'], ['v1', 'v2', 'v3'])
    #     batch = fd.transform(self.df.sample(1000, replace=True))
    #     b = (batch == 'v1').sum(axis=0)
    #     assert binom_test(b[0], 1000, 0.33) > 0.01
    #     assert b[1] == 0
    #     assert b[2] == 0
    #     b = (batch == 'v2').sum(axis=0)
    #     assert binom_test(b[1], 1000, 0.33) > 0.01
    #     assert b[0] == 0
    #     assert b[2] == 0
    #     b = (batch == 'v3').sum(axis=0)
    #     assert binom_test(b[2], 1000, 0.33) > 0.01
    #     assert b[0] == 0
    #     assert b[1] == 0
    #
    # def test_multiple_feature_drop(self):
    #     fd = FeatureDropout(1., ['var1', 'var2', 'label'], '', col_probs=[.5, .3, .2], n_probs=[.7, .3])
    #     batch = fd.transform(self.df.sample(1000, replace=True))
    #     b = (batch == '').sum(axis=1).value_counts().sort_index().tolist()
    #     c, p = chisquare(b, [700, 300])
    #     assert p > 0.01
    #
    # def test_parameter_error_handling(self):
    #     # column name is not str
    #     with pytest.raises(ValueError):
    #         fd = FeatureDropout(1., 1, 'v1')
    #     with pytest.raises(ValueError):
    #         fd = FeatureDropout(1., ['var1', 'var2', 1], ['v1', 'v2', 'v3'])
    #     # drop_values and cols are same length
    #     with pytest.raises(ValueError):
    #         fd = FeatureDropout(1., ['var1', 'var2', 'label'], ['v1', 'v2'])
    #     with pytest.raises(ValueError):
    #         fd = FeatureDropout(1., ['var1', 'var2'], ['v1', 'v2', 'v3'])
    #     with pytest.raises(ValueError):
    #         fd = FeatureDropout(1., 'var1', ['v1', 'v2', 'v3'])
    #     # col_probs is the same length as cols
    #     with pytest.raises(ValueError):
    #         fd = FeatureDropout(1., ['var1', 'var2', 1], ['v1', 'v2', 'v3'], col_probs=[.5, .5])
    #     with pytest.raises(ValueError):
    #         fd = FeatureDropout(1., 'var1', 'v1', col_probs=[.5, .5])
    #     # when single column is transformed, col_probs is not accepted
    #     with pytest.raises(ValueError):
    #         fd = FeatureDropout(1., 'var1', 'v1', col_probs=.5)
