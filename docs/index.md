# Overview

Keras batchflow is a batch generator framework for [Keras](https://keras.io). 
The framework is generating batches for keras fit_generator and predict_generator functions in all sorts of less 
standard scenarios: multi-input and multi-output scenarios, scenarios employing dynamic data augmentation with dependencies 
between variables, etc.

The framework bridges gaps between keras and other two core data science modules: pandas and sklearn. With it, you can
use pandas dataframe directly as a datasource for your keras model. You can use all breadth of standard sklearn encoders 
to transform columns of a dataframe into numpy arrays. 

Following keras's modular approach, keras batchflow is built in the same way. A batch generator the framework produces 
is actually 
a stack of few interchangeable objects selected from a library, just like keras's layers. Each selected object adds new
feature to the generator. All of this gives combinatorial strength to the framework which is capable covering a wide range of 
scenarios by just mixing library objects. 

# Quick taster example

Here is a quick example of what the framework can do:

```python
import pandas as pd
from sklearn import LabelEncoder, LabelBinarizer
from keras_batchflow.batch_generators import BatchGenerator

df = pd.DataFrame({
    'var1': ['Class 0', 'Class 1', 'Class 0', 'Class 2', 'Class 0', 'Class 1', 'Class 0', 'Class 2'],
    'var2': ['Green', 'Yellow', 'Red', 'Brown', 'Green', 'Yellow', 'Red', 'Brown'],
    'label': ['Leaf', 'Flower', 'Leaf', 'Branch', 'Green', 'Yellow', 'Red', 'Brown']
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
([(None, ), (None, )], (None, 3))
``` 

The first element is a x_structure and it is a list if two inputs. Both of them are outputs of LabelEncoders, that
return integer ids of categorical variables, hence only one dimension. The y_structure is a single output produced by 
one-hot encoder, hence 2 dimensions.

Now you can define a neural network and use above generator in `fit_generator` to train it:

```python
from keras.layers import Dense, Concatenate, Embedding, Input
from keras.models import Model

shapes = bg.shape
n_classes = bg.n_classes

var1_input = Input(batch_shape=shapes[0][0])
var2_input = Input(batch_shape=shapes[0][2])
var1_emb = Embedding(n_classes[0][0], 10)(var1_input)
var2_emb = Embedding(n_classes[0][1], 10)(var2_input)
features = Concatenate()([var1_emb, var2_emb])
# shapes[1] is (None, 3), so shapes[1][1] is just 3
classes = Dense(shapes[1][1], activation='softmax')(features)

model = Model([var1_input, var2_input], classes)

model.compile('adam', 'categorical_crossentropy')

model.fit_generator(bg)
```

# Installation

```shell script
pip install git+https://github.com/maxsch3/keras-batchflow.git
```
