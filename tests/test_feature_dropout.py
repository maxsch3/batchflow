import pytest
import pandas as pd
from scipy.stats import binom_test, chisquare
from keras_batchflow.base.batch_transformers import FeatureDropout


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
        fd = FeatureDropout([0., 1.], 'var1', drop_values='')
        batch = fd.transform(self.df)
        assert type(batch) == pd.DataFrame
        assert batch.shape == self.df.shape
        assert batch.columns.equals(self.df.columns)
        assert batch.index.equals(self.df.index)
        assert (batch['var1'] == '').all()

    def test_row_dist(self):
        fd = FeatureDropout([0.4, .6], 'var1', drop_values='')
        batch = fd.transform(self.df.sample(1000, replace=True))
        b = (batch['var1'] == '').sum()
        assert binom_test(b, 1000, 0.6) > 0.01

    def test_cols_dist(self):
        sample_size = 1000
        fd = FeatureDropout([0., 1.], ['var1', 'var2', 'label'], drop_values='', col_probs=[.5, .3, .2])
        batch = fd.transform(self.df.sample(sample_size, replace=True))
        b = (batch == '').sum(axis=0)
        c, p = chisquare(b, [sample_size * .53, sample_size * .3, sample_size * .17])
        assert p > 0.001

    def test_uniform_col_dist(self):
        sample_size = 1000
        fd = FeatureDropout([0., 1.], ['var1', 'var2', 'label'], drop_values='')
        batch = fd.transform(self.df.sample(sample_size, replace=True))
        b = (batch == '').sum(axis=0)
        c, p = chisquare(b, f_exp=[sample_size/3, sample_size/3, sample_size/3])
        assert p > 0.01

    def test_different_drop_values(self):
        fd = FeatureDropout([0., 1.], ['var1', 'var2', 'label'], drop_values=['v1', 'v2', 'v3'])
        batch = fd.transform(self.df.sample(1000, replace=True))
        b = (batch == 'v1').sum(axis=0)
        assert binom_test(b[0], 1000, 0.33) > 0.01
        assert b[1] == 0
        assert b[2] == 0
        b = (batch == 'v2').sum(axis=0)
        assert binom_test(b[1], 1000, 0.33) > 0.001
        assert b[0] == 0
        assert b[2] == 0
        b = (batch == 'v3').sum(axis=0)
        assert binom_test(b[2], 1000, 0.33) > 0.001
        assert b[0] == 0
        assert b[1] == 0

    def test_multiple_feature_drop(self):
        sample_size = 100
        fd = FeatureDropout([0., .7, .3], ['var1', 'var2', 'label'], drop_values='', col_probs=[.5, .3, .2])
        batch = fd.transform(self.df.sample(sample_size, replace=True))
        b = (batch == '').sum(axis=1).value_counts().sort_index().tolist()
        c, p = chisquare(b, [sample_size * .7, sample_size * .3])
        assert p > 0.001

    def test_parameter_error_handling(self):
        # column name is not str
        with pytest.raises(ValueError):
            fd = FeatureDropout([.9, .1], 1, 'v1')
        with pytest.raises(ValueError):
            fd = FeatureDropout([.9, .1], ['var1', 'var2', 1], ['v1', 'v2', 'v3'])
        # drop_values and cols are same length
        with pytest.raises(ValueError):
            fd = FeatureDropout([.9, .1], ['var1', 'var2', 'label'], drop_values=['v1', 'v2'])
        with pytest.raises(ValueError):
            fd = FeatureDropout([.9, .1], ['var1', 'var2'], drop_values=['v1', 'v2', 'v3'])
        with pytest.raises(ValueError):
            fd = FeatureDropout([.9, .1], 'var1', drop_values=['v1', 'v2', 'v3'])
        # col_probs is the same length as cols
        with pytest.raises(ValueError):
            fd = FeatureDropout([.9, .1], ['var1', 'var2', 1], drop_values=['v1', 'v2', 'v3'], col_probs=[.5, .5])
        with pytest.raises(ValueError):
            fd = FeatureDropout([.9, .1], 'var1', drop_values='v1', col_probs=[.5, .5])
        # when single column is transformed, col_probs is not accepted
        with pytest.raises(ValueError):
            fd = FeatureDropout([.9, .1], 'var1', drop_values='v1', col_probs=.5)
