from abc import abstractmethod
import pandas as pd


class IEncoderAdaptor:

    """
    This class is used for converting data between source data in pandas dataframe and encoders, which might accept
    different format types, e.g. numpy array or pandas series

    This is an interface class for defining 2 standard adapters for numpy-based and pandas-based encoders. It can
    also be used for defining your own class if needed.
    """

    @abstractmethod
    def transform(self, x: pd.Series):
        """
        This method converts data before sending to an encoder. Define format conversion here
        :param x:
        :return:
        """
        pass

    @abstractmethod
    def inverse_transform(self, x, dtype=None) -> pd.Series:
        """
        This method coverts data received from encoder back into pandas Series
        :param x: data in the format of encoder
        :param dtype: optional, target dtype for data in pandas Series object created. Normally this is an original dtype
        from original data. If not provided, the data type will be inferred by pandas
        :return: a pandas series object
        """
        pass
