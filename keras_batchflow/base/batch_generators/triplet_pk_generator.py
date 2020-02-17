import pandas as pd
import numpy as np
from .batch_generator import BatchGenerator


class TripletPKGenerator(BatchGenerator):

    """
    This class implements a batch generator for generic triplet network described in this
    [paper](https://arxiv.org/abs/1703.07737)
    TODO: add more details
    """

    def __init__(self,
                 data: pd.DataFrame,
                 triplet_label,
                 classes_in_batch,
                 samples_per_class,
                 x_structure,
                 y_structure=None,
                 **kwargs):
        self.triplet_label = triplet_label
        self.class_ref = data[self.triplet_label].value_counts()
        self.classes_in_batch = classes_in_batch
        self.sample_per_class = samples_per_class
        super().__init__(data, x_structure, y_structure, shuffle=False, **kwargs)

    def __len__(self):
        """
        Helps to determine number of batches in one epoch
        :return:
        n - number of batches
        """
        return int(np.ceil(self.data.shape[0] / (self.sample_per_class * self.classes_in_batch)))

    def _select_batch(self, index):
        """
        This method is the core of triplet PK batch generator
        """
        classes_selected = self.class_ref.sample(self.classes_in_batch).index.values
        batch = self.data.loc[self.data[self.triplet_label].isin(classes_selected), :].\
            groupby(self.triplet_label).apply(self._select_samples_for_class)
        return batch

    def _select_samples_for_class(self, df):
        if df.shape[0] <= self.sample_per_class:
            return df
        return df.sample(self.sample_per_class, replace=False)
