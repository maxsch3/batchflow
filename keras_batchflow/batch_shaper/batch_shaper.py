from sklearn.base import BaseEstimator, TransformerMixin
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
        if data_sample is not None:
            self.fit_shapes(data_sample)
        pass

    def transform(self, data: pd.DataFrame, **kwargs):
        return self.__walk(data, self.__transform_func, **kwargs)

    @property
    def shape(self):
        if self.measured_shape is None:
            raise RuntimeError('Error: shapes of the output are nor yet measured from transformers provided. '
                               'please use method fit_shapes before accessing shape property')
        return self.measured_shape

    def fit_shapes(self, data_sample):
        if type(data_sample) != pd.DataFrame:
            raise ValueError('Error:')
        self.measured_shape = self.__walk(data_sample, self.__shape_func)

    def __walk(self, data: pd.DataFrame, func, **kwargs):
        x = self.__walk_structure(data, self.x_structure, func)
        return_y = self.y_structure is not None
        if 'return_y' in kwargs:
            return_y = kwargs['return_y']
        if return_y:
            y = self.__walk_structure(data, self.y_structure, func)
            return x, y
        else:
            return x

    def __walk_structure(self, data: pd.DataFrame, struc, func):
        """This will call a func on tuples detected as leafs. For branches, it will call itself recursively until a
        leaf reached"""
        if type(struc) is list:
            ret = [self.__walk_structure(data, s, func) for s in struc]
            return ret
        elif type(struc) is tuple:
            if self.__is_leaf(struc):
                ret = func(data=data, leaf=struc)
                return ret
            else:
                ret = tuple([self.__walk_structure(data, s, func) for s in struc])
                return ret
        else:
            raise ValueError('Error: structure definition in {} class only supports lists and tuples, but {}'
                             'was found'.format(type(self).__name__, type(struc)))

    def __is_leaf(self, struc):
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

    def __check_leaf(self, data, leaf, calling_func):
        if not self.__is_leaf(leaf):
            raise RuntimeError('Error: method {}.{} only accepts leaf of a structure, but something'
                               ' else was provided'.format(type(self).__name__, calling_func))
        if (leaf[0] not in data.columns) & (leaf[0] is not None):
            raise KeyError('Error: column {} was not found in data provided'.format(leaf[0]))

    def __transform_func(self, data, leaf):
        self.__check_leaf(data, leaf, 'transform')
        if leaf[0] is None:
            return np.repeat(leaf[1], data.shape[0])
        if leaf[1] is None:
            return data[leaf[0]].values
        if not hasattr(leaf[1], 'transform'):
            raise ValueError('Error: transformer of class {} provided in structure definition has no '
                             ' \'{}\' method'.format(type(leaf[1]).__name__, 'transform'))
        try:
            x = getattr(leaf[1], 'transform')(data[leaf[0]])
        except ValueError as e:
            raise ValueError('Error: ValueError exception occured while calling {}.{} method. Most likely you used'
                             ' 2D transformer. At the moment, only 1D transformers are supported. Please use 1D '
                             'variant or use wrapper'.format(type(leaf[1]).__name__, 'transform'))
        except Exception as e:
            raise RuntimeError('Error: unknown error while calling transform method of {} class provided in '
                               'structure. Error was:'.format(type(leaf[1]).__name__, e))
        return x

    def __shape_func(self, data, leaf):
        """
        Not sure how to implement it yet. Maybe transform and then shape of the output. Maybe it is not needed at all
        :param leaf:
        :return:
        """
        x = self.__transform_func(data, leaf)
        if x.ndim == 1:
            return None,
        else:
            return None, x.shape[1]
