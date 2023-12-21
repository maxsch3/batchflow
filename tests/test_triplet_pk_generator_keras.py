import numpy as np
import pandas as pd
import pytest
import importlib
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
# import BatchGenerator from keras namespace


try:
    import keras
    from keras_batchflow.keras.batch_generators import TripletPKGenerator
    from keras_batchflow.base.batch_generators import TripletPKGenerator as BaseTripletPKGenerator
    from keras.utils import Sequence
except ImportError:
    pass


@pytest.mark.skipif(importlib.util.find_spec("keras") is None,
                    reason='Keras is not installed in this environment (not needed when testing tensorflow 2 )')
class TestTripletPKGeneratorKeras:

    data = None
    y1_enc = None
    y2_enc = None
    triplet_loss = None

    def setup_method(self):
        self.data = pd.DataFrame({'x1': np.random.uniform(size=1000),
                                  'x2': np.random.uniform(size=1000),
                                  'y1': np.random.choice([0, 1, 2, 3], 1000),
                                  'y2': np.random.choice(['a', 'b', 'c', 'd', 'e'], 1000)})
        self.y1_enc = LabelEncoder().fit(self.data['y1'])
        self.y2_enc = LabelEncoder().fit(self.data['y2'])
        self.triplet_loss = lambda x, y: tf.contrib.losses.metric_learning.triplet_semihard_loss(x, y, margin=1.0)

    def test_mro(self):
        mro = TripletPKGenerator.mro()
        i1 = [i for i, x in enumerate(mro) if x == TripletPKGenerator]
        i2 = [i for i, x in enumerate(mro) if x == Sequence]
        assert len(i1) == 1
        assert len(i2) == 1
        # check that BaseBatch generator is resolved before SequenceTF
        assert i1 < i2

    def test_fit_generator_without_validation(self):
        # declare a batch generator
        # keras_bg = TripletPKGenerator(self.data,
        #                               triplet_label='y1',
        #                               classes_in_batch=3,
        #                               samples_per_class=10,
        #                               x_structure=('x1', None),
        #                               y_structure=('y1', self.y1_enc))
        # inp = keras.layers.Input(shape=(1,))
        # x = keras.layers.Dense(10, activation='relu')(inp)
        # x = keras.layers.Dense(5, use_bias=False, activation='linear')(x)
        # keras_model = keras.models.Model(inp, x)
        # keras_model.compile('adam', tf.contrib.losses.metric_learning.triplet_semihard_loss)
        # keras_model.fit_generator(keras_bg, epochs=1)
        pass
