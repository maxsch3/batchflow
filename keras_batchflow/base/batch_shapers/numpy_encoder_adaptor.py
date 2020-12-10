import pandas as pd
import numpy as np

from .encoder_adaptor import IEncoderAdaptor


class NumpyEncoderAdaptor(IEncoderAdaptor):

    """
    This is an adapter that converts data for Numpy based encoders.
    """

    def transform(self, x: pd.Series):
        """
        This method converts pandas Series object to a numpy array. It uses to_numpy method rather than values
        property because some Pandas data types do not exist in numpy (e.g. IntegerArray) and by using to_numpy,
        we involve pandas internal conversion
        :param x:
        :return:
        """
        return x.to_numpy()

    def inverse_transform(self, x, dtype=None) -> pd.Series:
        """
        Both input and output are pandas Series here. THis method will only check that encoder indeed returns
        pandas Series and will make sure the data type is correct
        :param x:
        :param dtype:
        :return:
        """
        if not isinstance(x, np.ndarray):
            raise TypeError(f"Error: the encoder is supposed to return numpy array, got {type(x)}")
        if x.ndim > 1:
            x = np.squeeze(x)
            if x.ndim > 1:
                raise ValueError(f"Error: the encoder is supposed to return 1D data. Got {x.ndim}D even after "
                                 f"squeezing")
        x = pd.Series(x, dtype=dtype)
        return x
