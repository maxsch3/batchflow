import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from batch_generator.triplet_pk_generator2d import TripletPKGenerator2D
from transformer.identity_transform import IdentityTransform


class TestTripletPKGenerator2D:

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
        tg = TripletPKGenerator2D(
            data=self.df,
            triplet_label=['label', 'var1'],
            classes_in_batch=2,
            samples_per_class=2,
            x_structure=('id', self.it),
            y_structure=('label', self.it)
        )
        batch = tg[0]
        assert type(batch) == tuple
        assert len(batch) == 2
        assert type(batch[0]) == list
        assert len(batch[0]) == 3
        if 'Flower' in batch[1]:
            assert batch[0][0].shape == (3,)
        else:
            assert batch[0][0].shape == (4,)
