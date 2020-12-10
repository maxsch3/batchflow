import numpy as np
import pandas as pd
from numbers import Number

from .encoder_adaptor import IEncoderAdaptor
from .numpy_encoder_adaptor import NumpyEncoderAdaptor
from .pandas_encoder_adaptor import PandasEncoderAdaptor


class VarShaper:

    """
    This class is a wrapper around encoder. It abstracts away the encoding of a single column into a
    model-ready data. This class is used in BatchShaper where all leaves (a tuple (column name, encoder))
    in a structure are replaced with equivalent VarShaper objects in a BatchShaper's constructor method
    After that, BatchShaper use this equivalent structure of VarShaper objects rather than original
    "human-readable" structure of tuples.

    This class is a backend class and you normally would not need to use it manually.

    It implements core interface functions like transform, inverse_transform,
    as well as additional metadata functions like, shape, n_classes, dtype, etc.
    """

    _dummy_constant_counter = 0

    def __init__(self, var_name, encoder, data_sample=None, encoder_adaptor=None):
        self._var_name = var_name
        # _name will be included in metadata for using in ML models, e.g. for naming input layers in Keras
        self._name = var_name
        self._encoder_adaptor = self._build_encoder_adaptor(encoder_adaptor)
        self._encoder = encoder
        self._class = self._self_classify(var_name, encoder)
        self._decoded_dtype, self._dtype = self._get_dtypes(data_sample)
        if self._class == "constant":
            # this is a class level counter to count all instances of constant class
            self._name = 'dummy_constant_' + str(self._dummy_constant_counter)
            type(self)._dummy_constant_counter += 1
        self._shape = self._get_shape(var_name, encoder, data_sample)
        self._n_classes = self._get_n_classes(encoder)

    def _build_encoder_adaptor(self, encoder_adaptor):
        """
        This method makes encoder adaptor class that utilises polymorphism to accommodate encoders that require
        different data types. The encoder adaptor instance takes care of
        :param encoder_adaptor: str ('numpy' or 'pandas') or a custom class derived from IEncoderAdaptor
        :return: instance of IEncoderAdaptor
        """
        if (encoder_adaptor == 'numpy') or (encoder_adaptor is None):
            adaptor = NumpyEncoderAdaptor()
        elif encoder_adaptor == 'pandas':
            adaptor = PandasEncoderAdaptor()
        elif isinstance(encoder_adaptor, IEncoderAdaptor):
            adaptor = encoder_adaptor
        else:
            raise TypeError(f"Error: The encoder adaptor must be a string ('numpy' or 'pandas') or an instance of a "
                            f"custom class derived from IEncoderAdaptor")
        return adaptor

    @staticmethod
    def _self_classify(var_name, encoder):
        """
        This method identifies what type of variable the class is responsible for:
        - encoded typical scenario when encoder is used to change the original value to some encoded representation
        - constant when variable is made up by the batch generator. This variable has constant value specified
         in encoder parameter. e.g. 0.
        - directly mapped when a value passed from a dataframe to the output without any changes
        :param var_name: a str or None. Str is used for encoded or directly mapped classes of variables. None is only
        used for constant
        :param encoder: None, scalar value or encoder class (any class having transform and inverse_transform methods)
        :return: onr of the following strings: "constant", "direct" or "encoder"
        """
        if var_name is None:
            if isinstance(encoder, Number) or isinstance(encoder, str):
                return "constant"
            else:
                raise ValueError(f"Error: for variables = None correct format of a structure element is "
                                 f"(None, any_constant), got (None, {encoder})")
        elif isinstance(var_name, str):
            if encoder is None:
                return "direct"
            if not hasattr(encoder, "transform"):
                raise ValueError(f"Error: encoder provided for column '{var_name}' has no 'transform' method")
            if not hasattr(encoder, "inverse_transform"):
                raise ValueError(f"Error: encoder provided for column '{var_name}' has no 'inverse_transform' method")
            return "encoder"
        else:
            raise ValueError(f"Error: variable name must be a str or None. Got {type(var_name)}")

    @property
    def metadata(self):
        metadata = {}
        metadata['name'] = self._name
        if np.isscalar(self._encoder):
            metadata['encoder'] = None
        else:
            metadata['encoder'] = self._encoder
        metadata['shape'] = self._shape
        metadata['dtype'] = self._dtype
        metadata['n_classes'] = self._n_classes
        return metadata

    @property
    def shape(self):
        return self._shape

    @property
    def n_classes(self):
        return self._n_classes

    def _get_shape(self, var_name, encoder, sample):
        if self._class in ["constant", "direct"]:
            return 1,
        else:
            if hasattr(encoder, 'shape'):
                return encoder.shape
            if sample is not None:
                x = self._reshape(self.transform(sample))
                if x.ndim == 1:
                    raise RuntimeError('This should not have happened. Please report this issue')
                return tuple(x.shape[1:])
            else:
                raise ValueError(f"Error: unable to determine encoded shape of variable {var_name} because neither "
                                 f"encoder {encoder} provided has attribute 'shape' nor data sample "
                                 f"provided to evaluate")

    def _get_dtypes(self, sample):
        """
        This method identifies dtype of encoded format and original decoded dtype of the variable stored in
        the original dataframe
        :param encoder: None, scalar constant or encoder object
        :param sample:
        :return: a tuple (original dtype, encoded dtype)
        """
        if self._class in ["encoder", "direct"]:
            if sample is None:
                raise ValueError(f"Error: Unable to determine encoded and original data types without a data sample. "
                                 f"Please provide a data sample in init")
        if self._class == "encoder":
            x = self._reshape(self.transform(sample))
            return sample[self._var_name].dtype, x.dtype
        elif self._class == "direct":
            return sample[self._var_name].dtype, sample[self._var_name].dtype
        elif self._class == "constant":
            original_dtype = self._encoder.dtype if hasattr(self._encoder, "dtype") else type(self._encoder)
            return original_dtype, np.array([self._encoder]).dtype
        else:
            RuntimeError(f"Error: the class type {self._encoder} is not supported in '_get_dtypes' method")

    def _get_n_classes(self, encoder):
        """
        Calculates number of classes provided by the encoder. This is required for creating embedding layer for
        the variable
        :param encoder: None, scalar constant or encoder object
        :return: integer scalar
        """
        if self._class in ["constant", "direct"]:
            return None
        elif self._class == "encoder":
            if hasattr(encoder, 'n_classes'):
                return encoder.n_classes
            if hasattr(encoder, 'classes_'):  # trying LabelEncoder compatible transformers
                return len(encoder.classes_)
            if hasattr(encoder, 'vocabulary_'):  # trying CountVectorizer  type of transformers
                return len(encoder.vocabulary_)
        else:
            raise RuntimeError(f"Error: '_class' is None. This should not have happened. Please report this error.")

    def transform(self, data):
        if self._class in ["encoder", "direct"]:
            if self._var_name not in data:
                raise KeyError(f"Error: column '{self._var_name}' is not available in data")
        #     if self._decoded_dtype is None:
        #         self._decoded_dtype = data[self._var_name].dtype
        # else:
        #     # if constant, pick dtype from self._encoder type
        #     self._decoded_dtype = self._encoder.dtype if hasattr(self._encoder, 'dtype') else type(self._encoder)
        if self._class == "encoder":
            # it has already been checked at init stage. It is redundant here
            # if not hasattr(self._encoder, 'transform'):
            #     raise ValueError(f"Error: encoders of class {type(self._encoder).__name__} provided in structure "
            #                      f"definition has no 'transform' method")
            encoder_input = self._encoder_adaptor.transform(data[self._var_name])
            try:
                x = self._encoder.transform(encoder_input)
            except ValueError as e:
                raise ValueError(f'Error: ValueError exception occured while calling '
                                 f'{type(self._encoder).__name__}.transform method. Most likely you used'
                                 f' 2D encoders. At the moment, only 1D transformers are supported. Please use 1D '
                                 f'variant or use wrapper. The error was: {e}')
            # except Exception as e:
            #     raise RuntimeError(f'Error: unknown error while calling transform method of '
            #                        f'{type(self._encoder).__name__} class provided in structure. The error was: {e}')
        elif self._class == "constant":
            x = np.repeat(self._encoder, data.shape[0])
        elif self._class == "direct":
            x = data[self._var_name].to_numpy()
        else:
            raise RuntimeError('Error: this should not have happened. Maybe it needs to be reported')
        # if self._dtype is None:
        #     self._dtype = x.dtype
        return self._reshape(x)

    def inverse_transform(self, df, encoded_data):
        """
        This method is used for converting encoded data returned by a predicting model back into a dataframe.
        :param df: A dataframe into which the decoded data are collected
        :param encoded_data: a numpy array which is returned by an ML model
        :return: None
        """
        if self._class == "constant":
            return
        if self._decoded_dtype is None:
            raise RuntimeError(f"Error: original data type is undefined. Please call transform before invoking "
                               f"inverse_transform.")
        if self._class == "direct":
            # This covers the case when a variable is not encoded and passed to and from X,Y structure without
            # changes. These cases have structure entry like this ('col_name', None)
            df[self._var_name] = pd.Series(np.squeeze(encoded_data), dtype=self._decoded_dtype)
        elif self._class == "encoder":
            # it has already been checked at init stage. It is redundant here
            # if not hasattr(self._encoder, "inverse_transform"):
            #     raise ValueError(f"Error: encoder provided for column '{self._var_name}' has no "
            #     "'inverse_transform' method")
            it = self._encoder_adaptor.inverse_transform(self._encoder.inverse_transform(encoded_data),
                                                         dtype=self._decoded_dtype)
            df[self._var_name] = it

    def _reshape(self, x: np.ndarray):
        if x.ndim == 1:
            return np.expand_dims(x, axis=-1)
        return x
