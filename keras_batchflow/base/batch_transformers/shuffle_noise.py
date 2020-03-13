from .base_random_cell import BaseRandomCellTransform
import numpy as np
import pandas as pd


class ShuffleNoise(BaseRandomCellTransform):

    def _make_augmented_version(self, batch):
        batch1 = batch.apply(lambda x: x.sample(frac=1))
        return batch1