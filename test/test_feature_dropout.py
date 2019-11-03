import pytest
import pandas as pd
import numpy as np
from scipy.stats import binom_test, chisquare
from keras_batchflow.batch_generator import BatchGenerator
from keras_batchflow.batch_transformer import FeatureDropout
from sklearn.preprocessing import LabelEncoder, LabelBinarizer, OneHotEncoder


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
        fd = FeatureDropout(1., 'var1', '')
        batch = fd.transform(self.df)
        assert type(batch) == pd.DataFrame
        assert batch.shape == self.df.shape
        assert batch.columns.equals(self.df.columns)
        assert batch.index.equals(self.df.index)
        assert (batch['var1'] == '').all()

    def test_row_dist(self):
        fd = FeatureDropout(.6, 'var1', '')
        batch = fd.transform(self.df.sample(1000, replace=True))
        b = (batch['var1'] == '').sum()
        assert binom_test(b, 1000, 0.6) > 0.01

    def test_cols_dist(self):
        fd = FeatureDropout(1., ['var1', 'var2', 'label'], '', col_probs=[.5, .3, .2])
        batch = fd.transform(self.df.sample(1000, replace=True))
        b = (batch == '').sum(axis=0)
        c, p = chisquare(b, [500, 300, 200])
        assert p > 0.01

    def test_uniform_col_dist(self):
        fd = FeatureDropout(1., ['var1', 'var2', 'label'], '')
        batch = fd.transform(self.df.sample(1000, replace=True))
        b = (batch == '').sum(axis=0)
        c, p = chisquare(b, [333, 333, 333])
        assert p > 0.01

    def test_different_drop_values(self):
        fd = FeatureDropout(1., ['var1', 'var2', 'label'], ['v1', 'v2', 'v3'])
        batch = fd.transform(self.df.sample(1000, replace=True))
        b = (batch == 'v1').sum(axis=0)
        assert binom_test(b[0], 1000, 0.33) > 0.01
        assert b[1] == 0
        assert b[2] == 0
        b = (batch == 'v2').sum(axis=0)
        assert binom_test(b[1], 1000, 0.33) > 0.01
        assert b[0] == 0
        assert b[2] == 0
        b = (batch == 'v3').sum(axis=0)
        assert binom_test(b[2], 1000, 0.33) > 0.01
        assert b[0] == 0
        assert b[1] == 0

