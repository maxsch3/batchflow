from tensorflow.keras.utils import Sequence as SequenceTF
import pandas as pd
from keras_batchflow.base.batch_generators import TripletPKGenerator as BaseTripletPKGenerator


class TripletPKGenerator(BaseTripletPKGenerator, SequenceTF):

    def __init__(self,
                 data: pd.DataFrame,
                 triplet_label,
                 classes_in_batch,
                 samples_per_class,
                 x_structure,
                 **kwargs):
        super(TripletPKGenerator, self).__init__(data, triplet_label, classes_in_batch,  samples_per_class,
                                                 x_structure, **kwargs)
