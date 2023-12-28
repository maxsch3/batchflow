import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from keras_batchflow.base.batch_generators import TripletPKGenerator
from keras_batchflow.base.batch_transformers import BatchTransformer


class TestTripletPKGenerator:

    df = None
    le = LabelEncoder()
    lb = LabelBinarizer()

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
            x_structure=('id', None),
            y_structure=('label', None)
        )
        assert len(tg) == 2
        batch = tg[0]
        assert type(batch) == tuple
        assert len(batch) == 2
        assert type(batch[0]) == np.ndarray
        assert batch[0].ndim == 2
        assert type(batch[1]) is np.ndarray
        if 'Flower' in batch[1]:
            assert batch[0].shape == (4, 1)
            assert batch[1].shape == (4, 1)
        else:
            assert batch[0].shape == (6, 1)
            assert batch[1].shape == (6, 1)
        batch = tg[1]
        if 'Flower' in batch[1]:
            assert batch[0].shape == (4, 1)
            assert batch[1].shape == (4, 1)
        else:
            assert batch[0].shape == (6, 1)
            assert batch[1].shape == (6, 1)

    def test_kwargs_pass_to_parent(self):

        class TransparentTransform(BatchTransformer):

            def transform(self, batch):
                return batch

            def inverse_transform(self, batch):
                return batch

        bt = TransparentTransform()
        tg = TripletPKGenerator(
            data=self.df,
            triplet_label='label',
            classes_in_batch=2,
            samples_per_class=3,
            batch_transforms=[bt],
            x_structure=('id', None),
            y_structure=('label', None)
        )
        pass



