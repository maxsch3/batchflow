# from sklearn.base import BaseEstimator, TransformerMixin
from inspect import isclass
import pandas as pd
import numpy as np


class BatchShaper:

    """
    a class that combined one or several sklearn-compatible transformers into horizontal structure

    I will support transform and inverse transform functions, but it will not do fit and fit_transform.
    It will assume that transforms used are already fitted. This is because transformers have to be fit
    on whole dataset that includes all labels, whilst the batch generator framework will usually be used
    with train/test splitted datasets
    """

    def __init__(self, x_structure, y_structure=None, data_sample=None):
        self.x_structure = x_structure
        self.y_structure = y_structure
        self.measured_shape = None
        self.__dummy_constant_counter = 0
        if data_sample is not None:
            self.fit_shapes(data_sample)
        pass

    def transform(self, data: pd.DataFrame, **kwargs):
        return self._walk(data, self._transform_func, **kwargs)

    def inverse_transform(self, y):
        df = pd.DataFrame()
        self._zipwalk_structure(df, self.y_structure, y, self._inverse_transform_func)
        return df

    @property
    def shape(self):
        if self.measured_shape is None:
            raise RuntimeError('Error: shapes of the output are nor yet measured from transformers provided. '
                               'please use method fit_shapes before accessing shape property')
        return self.measured_shape

    @property
    def n_classes(self):
        return self._walk(pd.DataFrame(), self._n_classes_func)

    def get_metadata(self, data_sample):
        self.__dummy_constant_counter = 0
        return self._walk(data_sample, self._gather_metadata_func)

    def fit_shapes(self, data_sample):
        if type(data_sample) != pd.DataFrame:
            raise ValueError('Error:')
        self.measured_shape = self._walk(data_sample, self._shape_func)

    def _walk(self, data: pd.DataFrame, func, **kwargs):
        x = self._walk_structure(data, self.x_structure, func)
        return_y = self.y_structure is not None
        if 'return_y' in kwargs:
            return_y = kwargs['return_y']
        if return_y:
            y = self._walk_structure(data, self.y_structure, func, **kwargs)
            return x, y
        else:
            return x

    def _walk_structure(self, data: pd.DataFrame, struc, func, **kwargs):
        """This will call a func on tuples detected as leafs. For branches, it will call itself recursively until a
        leaf reached"""
        if type(struc) is list:
            ret = [self._walk_structure(data, s, func, **kwargs) for s in struc]
            return ret
        elif type(struc) is tuple:
            if self._is_leaf(struc):
                ret = func(data=data, leaf=struc, **kwargs)
                return ret
            else:
                ret = tuple([self._walk_structure(data, s, func, **kwargs) for s in struc])
                return ret
        else:
            raise ValueError('Error: structure definition in {} class only supports lists and tuples, but {}'
                             'was found'.format(type(self).__name__, type(struc)))

    def _zipwalk_structure(self, data: pd.DataFrame, struc, struc_data, func, **kwargs):
        """This function works similar to _walk structure, with only one difference: it walks two structures
        in parallel (like zip). It is used in inverse transform logic where data returned by a model in the
        format of y_structure is walked together with y_structure itself so that relevant encoders are applied to
        correct parts of y_data"""
        if (type(struc) is list) & (type(struc_data) is list):
            ret = [self._zipwalk_structure(data, s, d, func, **kwargs) for s, d in zip(struc, struc_data)]
            return ret
        elif (type(struc) is tuple) & ((type(struc_data) is tuple) | (type(struc_data) is np.ndarray)):
            if self._is_leaf(struc):
                ret = func(data=data, struc_data=struc_data, leaf=struc, **kwargs)
                return ret
            else:
                ret = tuple([self._zipwalk_structure(data, s, d, func, **kwargs) for s, d in zip(struc, struc_data)])
                return ret
        else:
            raise ValueError('Error: structure definition and encoded data do not match'
                             .format(type(self).__name__, type(struc)))

    def _is_leaf(self, struc):
        if type(struc) is tuple:
            if len(struc) == 2:
                if type(struc[0]) is str:
                    if struc[1] is None:
                        return True
                    elif isclass(type(struc[1])):
                        return True
                    else:
                        raise ValueError('Error: a transformer must be an instance of a class on structure'
                                         ' definition in {} class'.format(type(self).__name__))
                elif struc[0] is None:
                    # scenario (None, 1.) when constant value is outputted
                    if np.isscalar(struc[1]):
                        return True
        return False

    def _check_leaf(self, data, leaf, calling_func, check_data=True):
        if not self._is_leaf(leaf):
            raise RuntimeError('Error: method {}.{} only accepts leaf of a structure, but something'
                               ' else was provided'.format(type(self).__name__, calling_func))
        if check_data:
            if (leaf[0] not in data.columns) & (leaf[0] is not None):
                raise KeyError('Error: column {} was not found in data provided'.format(leaf[0]))

    def _transform_func(self, data, leaf):
        self._check_leaf(data, leaf, 'transform')
        if leaf[0] is None:
            return np.repeat(leaf[1], data.shape[0])
        if leaf[1] is None:
            return data[leaf[0]].values
        if not hasattr(leaf[1], 'transform'):
            raise ValueError('Error: transformer of class {} provided in structure definition has no '
                             ' \'{}\' method'.format(type(leaf[1]).__name__, 'transform'))
        try:
            x = getattr(leaf[1], 'transform')(data[leaf[0]])
        except ValueError:
            raise ValueError('Error: ValueError exception occured while calling {}.{} method. Most likely you used'
                             ' 2D transformer. At the moment, only 1D transformers are supported. Please use 1D '
                             'variant or use wrapper'.format(type(leaf[1]).__name__, 'transform'))
        except Exception as e:
            raise RuntimeError('Error: unknown error while calling transform method of {} class provided in '
                               'structure. Error was:'.format(type(leaf[1]).__name__, e))
        return x

    def _inverse_transform_func(self, data, struc_data, leaf):
        # if 'y_data' not in kwargs:
        #     raise TypeError('_inverse_transform_func() is missing 1 requred argument: y_data')
        self._check_leaf(data, leaf, 'inverse_transform', check_data=False)
        if not hasattr(leaf[1], 'inverse_transform'):
            raise ValueError('Error: the transformer {} used for column {} has no inverse_transform method'
                             .format(type(leaf[1]).__name__, leaf[0]))
        it = leaf[1].inverse_transform(struc_data)
        data[leaf[0]] = it

    def _shape_func(self, data, leaf):
        """
        """
        self._check_leaf(data, leaf, 'shape')
        if leaf[0] is None:
            return None, 1
        if hasattr(leaf[1], 'shape'):
            return leaf[1].shape
        x = self._transform_func(data, leaf)
        if x.ndim == 1:
            return None, 1
        else:
            return (None,) + x.shape[1:]

    def _data_type_func(self, data, leaf):
        """
        """
        self._check_leaf(data, leaf, 'shape')
        x = self._transform_func(data, leaf)
        return x.dtype

    def _n_classes_func(self, data, leaf):
        # _check_leaf is not needed here because data is not used here. Moreover, data might be missing if called
        # from n_classes property function above
        if (leaf[0] is None) | (leaf[1] is None):
            return None
        if hasattr(leaf[1], 'n_classes'):
            return leaf[1].n_classes
        if hasattr(leaf[1], 'classes_'):    # trying LabelEncoder compatible transformers
            return len(leaf[1].classes_)
        if hasattr(leaf[1], 'vocabulary_'):   # trying CountVectorizer  type of transformers
            return len(leaf[1].vocabulary_)
        return None

    def _gather_metadata_func(self, data, leaf):
        self._check_leaf(data, leaf, 'gather_metadata')
        metadata = {}
        if leaf[0] is None:
            metadata['name'] = 'dummy_constant_' + str(self.__dummy_constant_counter)
            self.__dummy_constant_counter += 1
        else:
            metadata['name'] = leaf[0]
        metadata['encoder'] = leaf[1]
        if np.isscalar(leaf[1]):
            metadata['encoder'] = None
        metadata['shape'] = self._shape_func(data, leaf)
        metadata['dtype'] = self._data_type_func(data, leaf)
        metadata['n_classes'] = self._n_classes_func(data, leaf)
        return metadata
