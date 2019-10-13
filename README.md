# batchflow
Batch generator framework for Keras. The framework is primarily for generating batches in all sorts of non-standard
scenarios, when data is too big to keep in memory, when a model has multiple inputs and outputs, etc.

It takes a standard pandas dataframe as an input along with a set of pre-fitted sklearn transformers. The whole 
dataframe is splitted in batches and transformers are applied to particular columns. The results are then packaged in
a keras-friendly format

The project is in active development and therefore the documentation is not yet in place. Please come back later. 

Meanwhile, quick tester example of what the framework is capable of

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
    y structure=('label', label_enc),
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
