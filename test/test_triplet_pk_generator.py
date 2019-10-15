import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from keras_batchflow.batch_generator.triplet_pk_generator import TripletPKGenerator
from keras_batchflow.transformer.identity_transform import IdentityTransform


class TestTripletPKGenerator:

    df = None
    le = LabelEncoder()
    lb = LabelBinarizer()
    it = IdentityTransform()

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
        tg = TripletPKGenerator(
            data=self.df,
            triplet_label='label',
            classes_in_batch=2,
            samples_per_class=3,
            x_structure=('id', self.it),
            y_structure=('label', self.it)
        )
        assert len(tg) == 2
        batch = tg[0]
        assert type(batch) == tuple
        assert len(batch) == 2
        assert type(batch[0]) == list
        assert len(batch[0]) == 2
        assert type(batch[0][0]) == np.ndarray
        assert type(batch[0][1]) == np.ndarray
        assert batch[0][0].ndim == 1
        if 'Flower' in batch[1]:
            assert batch[0][0].shape == (4,)
        else:
            assert batch[0][0].shape == (6,)
        assert batch[0][1].ndim == 1
        assert np.unique(batch[0][1]).tolist() == [0, 1]
        batch = tg[1]
        if 'Flower' in batch[1]:
            assert batch[0][0].shape == (4,)
        else:
            assert batch[0][0].shape == (6,)
        assert batch[0][1].ndim == 1
        assert np.unique(batch[0][1]).tolist() == [0, 1]



