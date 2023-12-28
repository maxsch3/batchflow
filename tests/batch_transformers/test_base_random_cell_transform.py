import pytest
import pandas as pd
import numpy as np
from scipy.stats import binom_test, chisquare
from keras_batchflow.base.batch_transformers import BaseRandomCellTransform, BatchFork


class LocalVersionTransform(BaseRandomCellTransform):
    """
    BaseRandomCellTransform raise NotImplemented error in below function which does not allow to test
    transform functionality. I'm re-defining this method here to be able to test it.
    """
    def _make_augmented_version(self, batch):
        batch = batch.copy()
        for c in self._cols:
            batch[c] = ''
        return batch


class TestTransformInt(BaseRandomCellTransform):
    """
    BaseRandomCellTransform raise NotImplemented error in below function which does not allow to test
    transform functionality. I'm re-defining this method here to be able to test it.
    """

    def _make_augmented_version(self, batch):
        batch = batch.copy()
        for c in self._cols:
            batch[c] = 0
        return batch


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
        rct = BaseRandomCellTransform([.9, .1], 'var1')
        rct = BaseRandomCellTransform((.9, .1), 'var1')
        rct = BaseRandomCellTransform((.9, .1), 'var1', col_probs=None)
        rct = BaseRandomCellTransform([.8, .1, .1], ['var1', 'var2'], [.5, .5])

    def test_parameter_error_handling(self):
        with pytest.raises(ValueError):
            # tests non numeric value in n_probs
            rct = BaseRandomCellTransform([.9, 'str'], 'var1')
        with pytest.raises(ValueError):
            # tests single value in n_probs
            rct = BaseRandomCellTransform([.9], 'var1')
        with pytest.raises(ValueError):
            # tests single scalar value in n_probs
            rct = BaseRandomCellTransform(.9, 'var1')
        with pytest.raises(ValueError):
            # tests numeric values in cols
            rct = BaseRandomCellTransform([.9, .1], 5)
        with pytest.raises(ValueError):
            # tests numeric values in cols
            rct = BaseRandomCellTransform([.9, .1], [5, 'd'])
        with pytest.raises(ValueError):
            # tests length check in cols and col_probs paramterers
            rct = BaseRandomCellTransform([.9, .1], 'var1', [.5, .1])
        with pytest.raises(ValueError):
            # tests length check in cols and col_probs paramterers
            rct = BaseRandomCellTransform([.9, .1], 'var1', [.5, .1])
        with pytest.raises(ValueError):
            # tests if single col_probs is not accepted
            rct = BaseRandomCellTransform([.9, .1], 'var1', col_probs=.5)

    def test_calculate_col_weigts(self):
        rct = BaseRandomCellTransform([.0, 1.], ['var1', 'var2'], [.5, .5])
        weights = rct._calculate_col_weights(np.array([.5, .5]))
        assert type(weights) == np.ndarray
        assert weights.ndim == 1
        assert weights.shape == (2,)
        assert weights.sum() > 0
        # test weights are always > 0
        assert all([rct._calculate_col_weights(np.random.uniform(size=(3,))).min() > 0 for _ in range(100)])
        # test weights are calculated for different lengths
        assert all([rct._calculate_col_weights(np.random.uniform(size=(i,))).min() > 0 for i in range(2, 10)])
        # test weights values
        probs = [.5, .4, .1]
        weights = rct._calculate_col_weights(np.array(probs))
        # this is an analytic formula used to calculate weights (see BaseRandomCellTransform API for details)
        assert all([np.abs((w/weights.sum() - p)) < .00001 for w, p in zip(weights, probs)])

    def test_col_distribution_mask(self):
        col_probs = [.5, .3, .2]
        cols = ['var1', 'var2', 'label']
        rct = BaseRandomCellTransform([.0, 1.], cols, col_probs)
        data = self.df.sample(10000, replace=True)
        mask = rct._make_mask(data)
        assert type(mask) == np.ndarray
        assert mask.shape == (data.shape[0], len(cols))
        # check if it is a proper one-hot encoding
        assert mask.sum() == data.shape[0]
        expected_counts = [5250, 3050, 1700]
        threshold = .0001
        # the counts do not make counts ideally to expected 5000, 3000, 2000
        c, p = chisquare(mask.sum(0), expected_counts)
        if p <= threshold:
            print(f'Error. looks like the column distribution {mask.sum(0)} is too far from expected '
                  f'{expected_counts}')
        assert p > threshold

    def test_col_distribution_output(self):
        col_probs = [.5, .3, .2]
        cols = ['var1', 'var2', 'label']
        rct = LocalVersionTransform([.0, 1.], cols, col_probs)
        data = self.df.sample(10000, replace=True)
        batch = rct.transform(data)
        assert isinstance(batch, pd.DataFrame)
        assert batch.shape == data.shape
        expected_counts = [5250, 3050, 1700]
        threshold = .0001
        # the counts do not make counts ideally to expected 5000, 3000, 2000
        c, p = chisquare((batch == '').sum(0), expected_counts)
        if p <= threshold:
            print(f"Error. looks like the column distribution {(batch == '').sum(0)} is too far from expected "
                  f"{expected_counts}")
        assert p > threshold


    def test_zero_mask(self):
        rct = BaseRandomCellTransform([1., 0.], 'var1')
        mask = rct._make_mask(self.df)
        assert mask.shape == (self.df.shape[0], 1)
        assert mask.sum() < 0.001

    def test_wrong_probs(self):
        rct = BaseRandomCellTransform([.9, .1], 'var1')
        with pytest.raises(ValueError):
            # tests error message if n_probs do not add up to 1
            rct = BaseRandomCellTransform([.9, .01], 'var1')

    def test_transform(self):
        ct = LocalVersionTransform([.0, 1.], 'var1')
        batch = ct.transform(self.df.copy())
        assert isinstance(batch, pd.DataFrame)
        assert batch.shape == self.df.shape
        assert not batch.equals(self.df)
        batch = self.df.copy()
        batch1 = ct.transform(batch)
        # test if transform does in-place transform
        assert batch1.equals(batch)
        assert (batch1['var1'] == '').all()
        assert (batch1['var2'] != '').all()
        assert (batch1['label'] != '').all()

    def test_transform_many_cols(self):
        ct = LocalVersionTransform([.0, 1.], cols=['var1', 'var2'])
        # make a bigger batch to make sure augmenatation will always be used on the batch
        # when batch is small there is a small chance the batch will sieve through without augmentation due to its
        # random nature
        seed_df = self.df.sample(100, replace=True)
        batch = ct.transform(seed_df.copy())
        assert isinstance(batch, pd.DataFrame)
        assert batch.shape == seed_df.shape
        assert not batch.equals(seed_df)
        batch = seed_df.copy()
        batch1 = ct.transform(batch)
        # test if transform does in-place transform
        assert batch1.equals(batch)
        assert (batch1['var1'] == '').any()
        assert (batch1['var2'] == '').any()
        assert (batch1['label'] != '').all()

    def test_transform_fork(self):
        batch = pd.concat([self.df]*2, axis=1, keys=['x', 'y'])
        ct = LocalVersionTransform([.0, 1.], cols='var1', data_fork='x')
        batch1 = ct.transform(batch.copy())
        assert isinstance(batch1, pd.DataFrame)
        assert batch1.shape == batch.shape
        assert not batch1.equals(batch)
        # test that x has been transformed
        assert not batch1['x'].equals(batch['x'])
        # test that y has not been transformers
        assert batch1['y'].equals(batch['y'])
        assert (batch1['x']['var1'] == '').all()
        assert (batch1['x']['var2'] != '').all()
        assert (batch1['x']['label'] != '').all()

    def test_transform_fork_many_cols(self):
        batch = pd.concat([self.df]*2, axis=1, keys=['x', 'y'])
        ct = LocalVersionTransform([0., 0., 1.], cols=['var1', 'var2'], data_fork='x')
        batch1 = ct.transform(batch.copy())
        assert isinstance(batch1, pd.DataFrame)
        assert batch1.shape == batch.shape
        assert not batch1.equals(batch)
        # test that x has been transformed
        assert not batch1['x'].equals(batch['x'])
        # test that y has not been transformers
        assert batch1['y'].equals(batch['y'])
        assert (batch1['x']['var1'] == '').all()
        assert (batch1['x']['var2'] == '').all()
        assert (batch1['x']['label'] != '').all()

    def test_non_numpy_dtype(self):
        """
        This test is to make sure the transform does not convert data to numpy behind the scenes, causing
        unpredictable dtype changes
        :return:
        """
        data = pd.DataFrame({'var1': np.random.randint(low=0, high=10, size=100)}).astype('Int64')
        data.iloc[0, 0] = None
        data_forked = BatchFork().transform(data.copy())
        ct = TestTransformInt([0., 1.], cols=['var1'], data_fork='x')
        data_transformed = ct.transform(data_forked)
        assert all(dt.name == 'Int64' for dt in data_transformed.dtypes)


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
