from .base_random_cell import BaseRandomCellTransform
import numpy as np
import pandas as pd


class ShuffleNoise(BaseRandomCellTransform):

    """A batch transformation for adding noise to data by randomly shuffling columns. The noise is added by mixing
    incoming batch with its shuffled version using mask:

    ```python
    batch = batch.mask(mask, shuffled_batch)
    ```

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

    """

    def _make_augmented_version(self, batch):
        augmented_batch = batch.apply(lambda x: x.sample(frac=1).values)
        return augmented_batch
