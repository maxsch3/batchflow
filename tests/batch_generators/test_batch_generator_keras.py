import numpy as np
import pandas as pd
import pytest
import importlib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
# import BatchGenerator from keras namespace


try:
    import keras
    from keras_batchflow.keras.batch_generators import BatchGenerator
    from keras_batchflow.base.batch_generators import BatchGenerator as BaseBatchGenerator
    from keras.utils import Sequence
except ImportError:
    pass


@pytest.mark.skipif(importlib.util.find_spec("keras") is None,
                    reason='Keras is not installed in this environment (not needed when testing tensorflow 2 )')
class TestBatchGeneratorKeras:

    data = None
    y1_enc = None
    y2_enc = None

    def setup_method(self):
        self.data = pd.DataFrame({'x1': np.random.uniform(size=1000),
                                  'x2': np.random.uniform(size=1000),
                                  'y1': np.random.choice([0, 1, 2, 3], 1000),
                                  'y2': np.random.choice(['a', 'b', 'c', 'd', 'e'], 1000)})
        self.y1_enc = LabelEncoder().fit(self.data['y1'])
        self.y2_enc = LabelEncoder().fit(self.data['y2'])

    def test_mro(self):
        mro = BatchGenerator.mro()
        i1 = [i for i, x in enumerate(mro) if x == BaseBatchGenerator]
        i2 = [i for i, x in enumerate(mro) if x == Sequence]
        assert len(i1) == 1
        assert len(i2) == 1
        # check that BaseBatch generator is resolved before SequenceTF
        assert i1 < i2

    def test_fit_without_validation(self):
        # declare a batch generator
        keras_bg = BatchGenerator(self.data, x_structure=('x1', None), y_structure=('y1', self.y1_enc))
        inp = keras.layers.Input(shape=(1,))
        x = keras.layers.Dense(4, activation='softmax')(inp)
        keras_model = keras.models.Model(inp, x)
        keras_model.compile('adam', 'sparse_categorical_crossentropy')
        keras_model.fit(keras_bg, epochs=1)

    def test_fit_with_validation(self):
        # declare a batch generator
        train, test = train_test_split(self.data, test_size=.2)
        train_bg = BatchGenerator(train, x_structure=('x1', None), y_structure=('y1', self.y1_enc))
        test_bg = BatchGenerator(test, x_structure=('x1', None), y_structure=('y1', self.y1_enc))
        inp = keras.layers.Input(shape=(1,))
        x = keras.layers.Dense(4, activation='softmax')(inp)
        keras_model = keras.models.Model(inp, x)
        keras_model.compile('adam', 'sparse_categorical_crossentropy')
        keras_model.fit(train_bg, epochs=1, validation_data=test_bg)

    def test_fit_multiple_epochs(self):
        # declare a batch generator
        keras_bg = BatchGenerator(self.data, x_structure=('x1', None), y_structure=('y1', self.y1_enc))
        inp = keras.layers.Input(shape=(1,))
        x = keras.layers.Dense(4, activation='softmax')(inp)
        keras_model = keras.models.Model(inp, x)
        keras_model.compile('adam', 'sparse_categorical_crossentropy')
        keras_model.fit(keras_bg, epochs=2)

    def test_multiple_inputs(self):
        keras_bg = BatchGenerator(self.data, x_structure=[('x1', None), ('x2', None)],
                                  y_structure=('y1', self.y1_enc))
        inp1 = keras.layers.Input(shape=(1,))
        inp2 = keras.layers.Input(shape=(1,))
        x = keras.layers.Concatenate()([inp1, inp2])
        x = keras.layers.Dense(4, activation='softmax')(x)
        keras_model = keras.models.Model([inp1, inp2], x)
        keras_model.compile('adam', 'sparse_categorical_crossentropy')
        keras_model.fit(keras_bg, epochs=2)

    def test_multiple_inputs_outputs(self):
        keras_bg = BatchGenerator(self.data, x_structure=[('x1', None), ('x2', None)],
                                  y_structure=[('y1', self.y1_enc), ('y2', self.y2_enc)])
        inp1 = keras.layers.Input(shape=(1,))
        inp2 = keras.layers.Input(shape=(1,))
        x = keras.layers.Concatenate()([inp1, inp2])
        y1 = keras.layers.Dense(4, activation='softmax')(x)
        y2 = keras.layers.Dense(5, activation='softmax')(x)
        keras_model = keras.models.Model([inp1, inp2], [y1, y2])
        keras_model.compile('adam', 'sparse_categorical_crossentropy')
        keras_model.fit(keras_bg, epochs=2)

    def test_predict_with_y(self):
        keras_bg = BatchGenerator(self.data, x_structure=('x1', None), y_structure=('y1', self.y1_enc))
        inp = keras.layers.Input(shape=(1,))
        x = keras.layers.Dense(4, activation='softmax')(inp)
        keras_model = keras.models.Model(inp, x)
        keras_model.compile('adam', 'sparse_categorical_crossentropy')
        pred = keras_model.predict(keras_bg, verbose=1)
        assert type(pred) == np.ndarray

    def test_predict_without_y(self):
        keras_bg = BatchGenerator(self.data, x_structure=('x1', None))
        inp = keras.layers.Input(shape=(1,))
        x = keras.layers.Dense(4, activation='softmax')(inp)
        keras_model = keras.models.Model(inp, x)
        keras_model.compile('adam', 'sparse_categorical_crossentropy')
        pred = keras_model.predict(keras_bg, verbose=1)
        assert type(pred) == np.ndarray
