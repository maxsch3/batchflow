

from .triplet_pk_generator import TripletPKGenerator

class TripletPKGenerator2D(TripletPKGenerator):

    def _add_local_labeller(self, x_structure):
        # triplet label is now a list of two columns
        ll = [(self.triplet_label[0], self.local_labeler), (self.triplet_label[1], self.local_labeler)]
        if type(x_structure) == list:
            return x_structure + ll
        else:
            return [x_structure] + ll

    def _select_batch(self, index):
        pass

    def _select_samples_for_class(self, df):
        pass