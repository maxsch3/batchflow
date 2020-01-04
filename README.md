# Keras-batchflow

Keras batchflow is a pipeline-friendly batch generator framework for [Keras](https://keras.io). 
The framework is generating batches for Keras's fit_generator and predict_generator functions in all sorts of less 
standard scenarios: multi-input and multi-output scenarios, scenarios employing dynamic data augmentation with 
dependencies between variables, etc.

The framework bridges gaps between keras, pandas and sklearn. With Keras batchflow you can
use pandas dataframe directly as a datasource for your keras model. You can use all breadth of standard sklearn encoders 
to transform columns of a dataframe into a numpy arrays. 

Read the documenatation [here](https://maxsch3.github.io/keras-batchflow/) 

Meanwhile, quick tester example of what the framework is capable of

```python
import pandas as pd
from sklearn import LabelEncoder, LabelBinarizer
from keras_batchflow.batch_generators import BatchGenerator

df = pd.DataFrame({
    'var1': ['Leaf', 'Flower', 'Leaf', 'Branch', 'Green', 'Yellow', 'Red', 'Brown'],
    'var2': ['Green', 'Yellow', 'Red', 'Brown', 'Green', 'Yellow', 'Red', 'Brown'],
    'label': ['Class 0', 'Class 1', 'Class 0', 'Class 2', 'Class 0', 'Class 1', 'Class 0', 'Class 2'],
})

#prefit sklearn encoders
var1_enc = LabelEncoder().fit(df['var1'])
var2_enc = LabelEncoder().fit(df['var2'])
label_enc = LabelBinarizer().fit(df['label'])

# define a batch generator
train_gen = BatchGenerator(
    df,
    x_structure=[('var1', var1_enc), ('var2', var2_enc)],
    y_structure=('label', label_enc),
    batch_size=4,
    train_mode=True
)
```

The generator returns batches of format (x_structure, y_structure) and the shape of the batches is:

```python
>>> train_gen.shape
([(1, ), (1, )], (3, ))
``` 

The first element is a x_structure and it is a list if two inputs. Both of them are outputs of LabelEncoders, that
return integer ids of categorical variables, hence the dimension has just 1 column. The y_structure is a 
single output produced by one-hot encoder, hence the dimension has 3 columns.

Now you can define a neural network and use above generator in `fit_generator` to train it

```python
from keras.layers import Dense, Concatenate, Embedding, Input, Lambda
from keras.models import Model
import keras.backend as K

metadata_x, metadata_y = bg.metadata

var1_input = Input(shape=metadata_x[0]['shape'])
var2_input = Input(shape=metadata_x[0]['shape'])
var1_emb = Embedding(metadata_x[0]['n_classes'], 10)(var1_input)
var2_emb = Embedding(metadata_x[1]['n_classes'], 10)(var2_input)
features = Concatenate()([var1_emb, var2_emb])
# features has dimensions (None, 1, 10), so I remove the second one  
features = Lambda(lambda x: K.squeeze(x, axis=1))(features)
classes = Dense(metadata_y['shape'][0], activation='softmax')(features)

model = Model([var1_input, var2_input], classes)

model.compile('adam', 'categorical_crossentropy')

model.fit_generator(bg)
```

# Installation 

```shell script
pip install git+https://github.com/maxsch3/keras-batchflow.git
```
