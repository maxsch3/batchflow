import pandas as pd

from .encoder_adaptor import IEncoderAdaptor


class PandasEncoderAdaptor(IEncoderAdaptor):

    """
    This is an adapter that converts data for Pandas based encoders.
    """

    def transform(self, x: pd.Series):
        """
        Because both input and output data is a pandas Series there will be no conversion
        :param x:
        :return:
        """
        return x

    def inverse_transform(self, x, dtype=None) -> pd.Series:
        """
        Both input and output are pandas Series here. THis method will only check that encoder indeed returns
        pandas Series and will make sure the data type is correct
        :param x:
        :param dtype:
        :return:
        """
        if not isinstance(x, pd.Series):
            raise TypeError(f"Error: the encoder is supposed to return Pandas Series object, got {type(x)}")
        if dtype is not None:
            if x.dtype != dtype:
                x = x.astype(dtype)
        return x
