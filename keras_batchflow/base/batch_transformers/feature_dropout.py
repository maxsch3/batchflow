import numpy as np
import pandas as pd
from .base_random_cell import BaseRandomCellTransform


class FeatureDropout(BaseRandomCellTransform):

    """ A batch transformation for randomly dropping values in the data to some pre-defined values

    **Parameters**

    - **n_probs** - a *list*, *tuple* or a *one dimensional numpy array* of probabilities $p_0, p_1, p_2, ... p_n$.
        $p_0$ is a probability for a row to have 0 augmented elements (no augmentation), $p_1$ - one random cells,
        $p_2$  - two random cells, etc. A parameter must have at least 2 values, scalars are not accepted
    - **cols** - a *list*, *tuple* or *one dimensional numpy array* of strings with columns names to be transformed
        Number of columns must be greater or equal the length of **n_probs** parameter simply because there must be
        enough columns to choose from to augment n elements in a row
    - **col_probs** - (optional) a *list*, *tuple* or *one dimensional numpy array* of floats
        $p_{c_0}, p_{c_1}, ... p_{c_k}$ where k is the number of columns specified in parameter cols.
        $p_{c_0}$ is the probability of column 0 from parameter *cols* to be selected in when only one column is picked
        for augmentation. $p_{c_1}$ is the same for column 1, etc. It is important to understand that when two or more
        columns, are picked for a row, actual frequencies of columns will drift towards equal distribution with every
        new item added. In a case when number of columns picked for augmentation reaches its max allowed value
        (number of columns available to choose from parameter **cols**), there will be no choice and the actual counts
        of columns will be equal. This means the actual distribution will turn into a uniform discrete distribution.
        **Default: None**
    - **drop_values** - (optional)  a *list*, *tuple* or *one dimensional numpy array* of values. If not set, `None`
       will be used for all columns. If single value is set, it will be used for all columns. If a list of items is
       set, it must be the same length as cols parameter. In this case, values specified in drop_values will be used
       to fill dropped values in corresponding columns


    """

    def __init__(self, n_probs, cols, col_probs=None, drop_values=None):
        super(FeatureDropout, self).__init__(n_probs, cols, col_probs)
        self._drop_values = self._validate_drop_values(drop_values)

    def _validate_drop_values(self, drop_values):
        if (type(drop_values) is str) or not hasattr(drop_values, '__iter__'):
            drop_values = [drop_values]
        elif type(drop_values) not in [tuple, list, np.ndarray]:
            raise ValueError('Error: parameter cols must be a single value or list, tuple, numpy array of values')
        dv = np.array(drop_values)
        if dv.ndim > 1:
            raise ValueError('Error: drop_values must be a vector of one dimension or a scalar value')
        if (len(dv) == 1) and (len(self._cols) > 1):
            dv = np.repeat(dv, len(self._cols))
        if len(dv) != len(self._cols):
            raise ValueError('Error: drop_values and cols parameters must have same dimensions')
        return dv

    def _make_augmented_version(self, batch):
        return pd.DataFrame(np.tile(self._drop_values, (batch.shape[0], 1)), columns=self._cols, index=batch.index)
