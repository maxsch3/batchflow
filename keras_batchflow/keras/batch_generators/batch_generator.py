from keras.utils import Sequence as SequenceKeras
import pandas as pd
from keras_batchflow.base.batch_generators import BatchGenerator as BaseBatchGenerator


class BatchGenerator(BaseBatchGenerator, SequenceKeras):

    def __init__(self, data: pd.DataFrame, x_structure, **kwargs):
        super(BatchGenerator, self).__init__(data, x_structure, **kwargs)
