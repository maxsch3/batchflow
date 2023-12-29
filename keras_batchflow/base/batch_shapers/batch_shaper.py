# from sklearn.base import BaseEstimator, TransformerMixin
from inspect import isclass
import pandas as pd
import numpy as np

from .var_shaper import VarShaper

class BatchShaper:

    """
    a class that combined one or several sklearn-compatible transformers into horizontal structure

    I will support transform and inverse transform functions, but it will not do fit and fit_transform.
    It will assume that transforms used are already fitted. This is because transformers have to be fit
    on whole dataset that includes all labels, whilst the batch generator framework will usually be used
    with train/test splitted datasets
    """

    def __init__(self, x_structure, y_structure=None, data_sample=None, multiindex_xy_keys=None,
                 encoder_adaptor=None):
        multiindex_xy_keys = ('x', 'y') if multiindex_xy_keys is None else multiindex_xy_keys
        self._validate_multiindex_xy_keys(x_structure, y_structure, multiindex_xy_keys)
        self.multiindex_xy_keys = multiindex_xy_keys
        self._encoder_adaptor = encoder_adaptor
        data_sample_x, data_sample_y = self._get_data_xy(data_sample)
        self.x_structure = self._create_shapers(structure=x_structure, data_sample=data_sample_x)
        self.y_structure = self._create_shapers(structure=y_structure, data_sample=data_sample_y)
        self.measured_shape = None
        self.__dummy_constant_counter = 0

    def transform(self, data: pd.DataFrame, **kwargs):
        return self._shape_batch(data, self._transform_func, **kwargs)

    def inverse_transform(self, y):
        df = pd.DataFrame()
        self._zipwalk_structure(df, self.y_structure, y, self._inverse_transform_func)
        return df

    @property
    def shape(self):
        return self._shape_batch(pd.DataFrame(), self._shape_func)

    @property
    def n_classes(self):
        return self._shape_batch(pd.DataFrame(), self._n_classes_func)

    @property
    def metadata(self):
        return self._shape_batch(pd.DataFrame(), self._metadata_func)

    def _validate_multiindex_xy_keys(self, x_structure, y_structure, multiindex_xy_keys):
        if type(multiindex_xy_keys) is not tuple:
            raise ValueError('Error: srtuct_index parameter must be a tuple')
        if (len(multiindex_xy_keys) < 2) and y_structure:
            raise ValueError(f'Error: when y_structure is not None, multiindex_xy_keys must have two values. '
                             f'got {len(multiindex_xy_keys)}')
        if len(multiindex_xy_keys) > 2:
            raise ValueError(f'Error: multiindex_xy_keys must not be longer than 2. Got {len(multiindex_xy_keys)}')
        if len(multiindex_xy_keys) != len(set(multiindex_xy_keys)):
            raise ValueError(f"Error: All elements of 'multiindex_xy_keys' parameter must be unique")

    def _create_shapers(self, structure, data_sample):
        if structure is None:
            return None
        return self._walk_structure(data_sample, structure, self._create_shaper_func)

    def _create_shaper_func(self, data, leaf, **kwargs):
        self._check_structure_leaf(leaf)
        return VarShaper(var_name=leaf[0], encoder=leaf[1], data_sample=data, encoder_adaptor=self._encoder_adaptor)

    def _shape_batch(self, data: pd.DataFrame, func, **kwargs):
        """
        This method forms a batch. Depending on the functions provided it can return different data shaped the same way:
         - batch-shaped data for keras and tensorflow fit/predict, e.g. ([ndarray, ndarray], ndarray)
         - component shapes structure, e.g. ([(None, 2), (None, 5)], (None, 2)) for the same example
         - number of classes for categorical inputs, e.g. ([10, 3], 2)
        :param data: pandas dataframe with the data
        :param func: a function defining the output (self._shape_func, self._transform_func, self._n_classes_func)
        :param kwargs:
        :return:
        """
        data_x, data_y = self._get_data_xy(data)
        x = self._walk_structure(data_x, self.x_structure, func)
        return_y = self.y_structure is not None
        if ('return_y' in kwargs) and (self.y_structure is not None):
            return_y = kwargs['return_y']
        if return_y:
            y = self._walk_structure(data_y, self.y_structure, func, **kwargs)
            return x, y
        else:
            return x,

    def _walk_structure(self, data: pd.DataFrame, struc, func, **kwargs):
        """This will call a func on tuples detected as leafs. For branches, it will call itself recursively until a
        leaf reached"""
        if self._is_leaf(struc):
            ret = func(data=data, leaf=struc, **kwargs)
            return ret
        elif type(struc) in [list, tuple]:
            ret = [self._walk_structure(data, s, func, **kwargs) for s in struc]
            if type(struc) is tuple:
                return tuple(ret)
            else:
                return ret
        else:
            raise ValueError('Error: structure definition in {} class only supports lists and tuples, but {}'
                             'was found'.format(type(self).__name__, type(struc)))

    def _zipwalk_structure(self, data: pd.DataFrame, struc, struc_data, func, **kwargs):
        """This function works similar to _walk_structure, with only one difference: it walks two structures
        in parallel (like zip). It is used in inverse transform logic where data returned by a model in the
        format of y_structure is walked together with y_structure itself so that relevant encoders are applied to
        correct parts of y_data"""
        if self._is_leaf(struc):
            ret = func(data=data, struc_data=struc_data, leaf=struc, **kwargs)
            return ret
        elif type(struc) in [list, tuple]:
            ret = [self._zipwalk_structure(data, s, d, func, **kwargs) for s, d in zip(struc, struc_data)]
            if type(struc) is tuple:
                return tuple(ret)
            else:
                return ret
        else:
            raise ValueError('Error: structure definition and encoded data do not match'
                             .format(type(self).__name__, type(struc)))

    def _is_leaf(self, struc):
        if isinstance(struc, VarShaper):
            return True
        if type(struc) is tuple:
            if len(struc) == 2:
                if type(struc[0]) is str:
                    if struc[1] is None:
                        return True
                    elif isclass(type(struc[1])):
                        return True
                    else:
                        raise ValueError(f'Error: a encoders must be an instance of a class on structure'
                                         f' definition in {type(self).__name__} class')
                elif struc[0] is None:
                    # scenario (None, 1.) when constant value is outputted
                    if np.isscalar(struc[1]):
                        return True
        return False

    def _check_var_shaper(self, shaper, calling_func):
        if not isinstance(shaper, VarShaper):
            raise RuntimeError(f'Error: method {type(self).__name__,}.{calling_func} only accepts '
                               f'VarShaper class objects but {type(shaper)} was provided')

    def _check_structure_leaf(self, structure):
        if not self._is_leaf(structure):
            raise ValueError(f"Error: class {type(self).__name__,} only accepts tuples (var_name, encoder), "
                             f"(var_name, None) or (None, constant) as leafs of a structure.")

    def _transform_func(self, data, leaf, **kwargs):
        self._check_var_shaper(leaf, 'transform')
        return leaf.transform(data)

    def _inverse_transform_func(self, data, struc_data, leaf: VarShaper):
        leaf.inverse_transform(df=data, encoded_data=struc_data)

    def _shape_func(self, data, leaf: VarShaper, **kwargs):
        """
        """
        return leaf.shape

    def _n_classes_func(self, data, leaf, **kwargs):
        return leaf.n_classes

    def _metadata_func(self, data, leaf: VarShaper, **kwargs):
        self._check_var_shaper(leaf, '_metadata_func')
        return leaf.metadata

    def _reshape(self, x: np.ndarray):
        if x.ndim == 1:
            return np.expand_dims(x, axis=-1)
        return x

    def _get_data_xy(self, data):
        if data is None:
            return None
        nlevels = data.columns.nlevels
        if nlevels == 1:
            return data, data
        elif nlevels == 2:
            # define generator that will
            if not all([name in data for name in self.multiindex_xy_keys]):
                raise KeyError(f"Error: one or more index, defined in multiindex_xy_keys parameter, was not found "
                               f"in the data. Please check the parameter, input data or batch transforms, such as "
                               f"BatchFork, 'that might add the required indices. If you used BatchFork, "
                               f"please check that its parameter 'levels' is aligned with parameter "
                               f"'multiindex_xy_keys' of the BatchShaper.")
            return tuple([data[name] for name in self.multiindex_xy_keys])
