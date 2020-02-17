import pandas as pd
import numpy as np
import pytest
import scipy.stats as stats
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from keras_batchflow.base.batch_generators import TripletPKGenerator2D
from keras_batchflow.base.encoders import IdentityEncoder


@pytest.mark.skip(reason="2D version of generator is to be depreciated")
class TestTripletPKGenerator2D:

    df = None
    le = LabelEncoder()
    lb = LabelBinarizer()
    it = IdentityEncoder()

    def setup_method(self):
        self.df = pd.DataFrame({
            'id': [0, 1, 2, 3, 4, 5, 6, 7, 8],
            'var1': ['Class 0', 'Class 1', 'Class 0', 'Class 2', 'Class 1', 'Class 2', 'Class 1', 'Class 2', 'Class 1'],
            'var2': ['Green', 'Yellow', 'Red', 'Brown', 'Green', 'Yellow', 'Red', 'Brown', 'Red'],
            'label': ['Leaf', 'Flower', 'Leaf', 'Branch', 'Leaf', 'Branch', 'Leaf', 'Branch', 'Leaf']
        })
        self.le.fit(self.df['label'])
        self.lb.fit(self.df['var1'])

    def teardown_method(self):
        pass

    def test_basic(self):
        tg = TripletPKGenerator2D(
            data=self.df,
            triplet_label=['label', 'var1'],
            classes_in_batch=2,
            samples_per_class=2,
            x_structure=('id', None),
            y_structure=[('label', None), ('var1', None)]
        )
        batch = tg[0]
        assert type(batch) == tuple
        assert len(batch) == 2
        assert type(batch[0]) == list
        assert len(batch[0]) == 3
        batch_stats = [self._batch_stats(tg[0][1]) for _ in range(100)]
        # out of two outputs, one was leading and have 2 unique values
        assert all([any([c == 2 for c in b]) for b in batch_stats])
        batch_stats = pd.DataFrame(batch_stats)
        # also each non-leading column can have 1, 2 or 3 unique items therefore, in 100 samples
        # all of them are expected. Since leading column for sampling is chosen randomly in each
        # batch, above will be valid for both columns.
        assert len(batch_stats.iloc[:, 0].value_counts()) > 1
        assert len(batch_stats.iloc[:, 1].value_counts()) > 1
        _, p = stats.chisquare(batch_stats.iloc[:, 0].value_counts(), batch_stats.iloc[:, 0].value_counts())
        assert np.abs(1. - p) < 0.01

    @staticmethod
    def _batch_stats(batch):
        return tuple([len(np.unique(np.array(b))) for b in batch])
