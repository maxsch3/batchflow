from keras.utils import Sequence
import pandas as pd
from keras_batchflow.base.batch_generators import TripletPKGenerator as BaseTripletPKGenerator


class TripletPKGenerator(BaseTripletPKGenerator, Sequence):

    def __init__(self,
                 data: pd.DataFrame,
                 triplet_label,
                 classes_in_batch,
                 samples_per_class,
                 x_structure,
                 **kwargs):
        super(TripletPKGenerator, self).__init__(data, triplet_label, classes_in_batch,  samples_per_class,
                                                 x_structure, **kwargs)
