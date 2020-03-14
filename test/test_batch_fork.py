import pandas as pd
import numpy as np
from keras_batchflow.base.batch_transformers import BatchFork


class TestBatchFork:

    df = None

    def setup_method(self):
        self.df = pd.DataFrame({
            'var1': ['Class 0', 'Class 1', 'Class 0', 'Class 2', 'Class 0', 'Class 1', 'Class 0', 'Class 2'],
            'var2': ['Green', 'Yellow', 'Red', 'Brown', 'Green', 'Yellow', 'Red', 'Brown'],
            'label': ['Leaf', 'Flower', 'Leaf', 'Branch', 'Green', 'Yellow', 'Red', 'Brown']
        })

    def test_basic(self):
        bf = BatchFork()
        batch = bf.transform(self.df)
        assert len(batch.columns.names) == 2
        assert len(batch.index.names) == 1
        assert batch.columns.get_level_values(0).drop_duplicates().tolist() == ['x', 'y']
        assert batch['x'].shape == self.df.shape
        assert batch.shape == (self.df.shape[0], self.df.shape[1]*2)
        assert (batch.index == self.df.index).all()
        assert (batch['x'].columns == self.df.columns).all()
        assert batch['x'].equals(batch['y'])
        # check that forks are independent (can change independently)
        batch.loc[:, ('x', 'var1')] = 'Mod'
        assert (batch['x'].loc[:, 'var1'] == 'Mod').all()
        assert (batch['y'].loc[:, 'var1'] != 'Mod').all()
