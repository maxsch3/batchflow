import pandas as pd
import numpy as np
from .triplet_pk_generator import TripletPKGenerator


class TripletPKGenerator2D(TripletPKGenerator):

    def _add_local_labeller(self, x_structure, data):
        # triplet label is now a list of two columns
        self.class_ref = pd.crosstab(data[self.triplet_label[0]], data[self.triplet_label[1]])
        ll = [(self.triplet_label[0], self.local_labeler), (self.triplet_label[1], self.local_labeler)]
        if type(x_structure) == list:
            return x_structure + ll
        else:
            return [x_structure] + ll

    def _select_batch(self, index):
        """
        This method is the core of triplet PK batch generator
        """
        swap = np.random.binomial(1, 0.5, 1)
        if swap:
            triplet_labels = self.triplet_label[::-1]
            # sample along columns of crosstab
            classes_selected = self.class_ref.T.sample(self.classes_in_batch)
        else:
            triplet_labels = self.triplet_label
            classes_selected = self.class_ref.sample(self.classes_in_batch)
        classes1_selected = classes_selected.index.values
        classes2_selected = classes_selected.loc[:, classes_selected.sum() > 0].columns.values
        batch = self.data.loc[self.data[triplet_labels[0]].isin(classes1_selected), :].\
            groupby(triplet_labels[0]).apply(lambda x: self._select_samples_for_class(x,
                                                                                      classes2_selected,
                                                                                      triplet_labels[1]))
        return batch

    def _select_samples_for_class(self, df, classes2_selected, triplet_label):
        batch = df.loc[df[triplet_label].isin(classes2_selected), :]
        if batch.shape[0] <= self.sample_per_class:
            return batch
        return batch.sample(self.sample_per_class, replace=False)
