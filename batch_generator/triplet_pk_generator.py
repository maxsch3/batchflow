import pandas as pd
import numpy as np
from .batch_generator import BatchGenerator
from transformer.triplet_pk_batch_labeler import TripletPKBatchLabeler


class TripletPKGenerator(BatchGenerator):

    """
    This class implements a batch generator for generic triplet network described in this
    [paper](https://arxiv.org/abs/1703.07737)
    TODO: add more details
    """

    def __init__(self,
                 data: pd.DataFrame,
                 triplet_label: str,
                 classes_in_batch,
                 samples_per_class,
                 x_structure,
                 y_structure=None
                 ):
        self.triplet_label = triplet_label
        self.classes_in_batch = classes_in_batch
        self.sample_per_class = samples_per_class
        self.local_labeler = TripletPKBatchLabeler()
        triplet_x_structure = self._add_local_labeller(x_structure)
        self.class_list = data[triplet_label].value_counts()
        super().__init__(data, triplet_x_structure, y_structure, shuffle=False)

    def _add_local_labeller(self, x_structure):
        ll = (self.triplet_label, self.local_labeler)
        if type(x_structure) == list:
            return x_structure + [ll]
        else:
            return [x_structure, ll]

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
        classes_selected = self.class_list.sample(self.classes_in_batch).index.values
        batch = self.data.loc[self.data[self.triplet_label].isin(classes_selected), :]
        batch = self.data.loc[self.data[self.triplet_label].isin(classes_selected), :].\
            groupby(self.triplet_label).apply(self._select_samples_for_class)
        return batch

    def _select_samples_for_class(self, df):
        if df.shape[0] <= self.sample_per_class:
            return df
        return df.sample(self.sample_per_class, replace=False)
