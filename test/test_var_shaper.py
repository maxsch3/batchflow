import pandas as pd
import numpy as np
import pytest

from sklearn.preprocessing import LabelEncoder, LabelBinarizer, OneHotEncoder
from keras_batchflow.base.batch_shapers.var_shaper import VarShaper


class A:
    @property
    def shape(self):
        return 11,

    def transform(self, data):
        return data


class B:

    def transform(self, data):
        return np.zeros((data.shape[0], 15), dtype=np.int8)


class TestVarShaper:

    df = None
    le = LabelEncoder()
    lb = LabelBinarizer()
    oh = OneHotEncoder()

    def setup_method(self):
        self.df = pd.DataFrame({
            'var1': ['Class 0', 'Class 1', 'Class 0', 'Class 2'],
            'var2': ['Green', 'Yellow', 'Red', 'Brown'],
            'var3': [0, 1, 0, 2],
            'label': ['Leaf', 'Flower', 'Leaf', 'Branch']
        })
        self.le.fit(self.df['label'])
        self.oh.fit(self.df[['var1', 'var2']])
        self.lb.fit(self.df['var1'])

    @pytest.mark.dependency(name='test_self_classify')
    def test_self_classify(self):
        vs = VarShaper('label', self.le, sample=self.df)
        assert vs._class == "encoder"
        vs = VarShaper(None, 1.)
        assert vs._class == "constant"
        vs = VarShaper('label', None)
        assert vs._class == "direct"
        with pytest.raises(ValueError):
            vs = VarShaper('label', 1.)
        with pytest.raises(ValueError):
            vs = VarShaper(None, self.le)
        with pytest.raises(ValueError):
            vs = VarShaper(None, None)
        with pytest.raises(ValueError):
            vs = VarShaper('label', np.array([1, 3]))

    @pytest.mark.dependency(name='test_get_shape', depends=['test_self_classify'])
    def test_get_shape(self):
        a = A()
        # test encoded element when encoder provides the shape
        vs = VarShaper('label', a)
        assert vs._shape == (11,)
        # test when shape is not provided by an encoder but sample is available
        b = B()
        vs = VarShaper('label', b, sample=self.df)
        assert vs._shape == (15,)
        # check that init will fail if sample is missing and sample property is not available
        with pytest.raises(ValueError):
            vs = VarShaper('label', b)
        # test one-hot encoded
        vs = VarShaper('var1', self.lb, sample=self.df)
        assert vs._shape == (3,)

    @pytest.mark.dependency(name='test_transform_encoder', depends=['test_get_shape'])
    def test_transform_encoder(self):
        vs = VarShaper('label', self.le, sample=self.df)
        tr = vs.transform(self.df)
        assert type(tr) == np.ndarray
        assert tr.shape == (self.df.shape[0], 1)
        assert tr.dtype.kind == "i"
        # test one-hot encoded data
        vs = VarShaper('var1', self.lb, sample=self.df)
        tr = vs.transform(self.df)
        assert type(tr) == np.ndarray
        assert tr.shape == (self.df.shape[0], 3)
        assert tr.dtype.kind == "i"

    @pytest.mark.dependency(name='test_transform_direct', depends=['test_get_shape'])
    def test_transform_direct(self):
        vs = VarShaper('var2', None)
        tr = vs.transform(self.df)
        assert tr.shape == (self.df.shape[0], 1)
        assert tr.dtype.kind == "O"

    @pytest.mark.dependency(name='test_transform_constant', depends=['test_get_shape'])
    def test_transform_constant(self):
        # test constant mode and dtypes
        vs = VarShaper(None, 1.)
        tr = vs.transform(self.df)
        assert tr.shape == (self.df.shape[0], 1)
        assert tr.dtype.kind == "f"
        vs = VarShaper(None, 1)
        tr = vs.transform(self.df)
        assert tr.dtype.kind == "i"

    @pytest.mark.dependency(name='test_init', depends=['test_transform_encoder',
                                                       'test_transform_direct',
                                                       'test_transform_constant'])
    def test_init(self):
        """
        Tests init procedure
        """
        vs = VarShaper('label', self.le, sample=self.df)
        assert vs._class == "encoder"
        assert vs._n_classes == len(self.le.classes_)
        assert vs._var_name == 'label'
        assert vs._encoder == self.le
        assert vs._dtype.kind == 'i'
        assert vs._decoded_dtype.kind == self.df[vs._var_name].dtype

    @pytest.mark.dependency(name='test_metadata', depends=['test_init'])
    def test_metadata(self):
        """
        This checks that metadata returns correct structure:
        - all fields needed are included
        - all values are correct
        """
        VarShaper._dummy_constant_counter = 0
        vs = VarShaper('label', self.le, sample=self.df)
        md = vs.metadata
        assert type(md) == dict
        assert "shape" in md
        assert md['shape'] == vs._shape
        assert "name" in md
        assert md['name'] == vs._var_name
        assert "encoder" in md
        assert md['encoder'] == vs._encoder
        assert "dtype" in md
        assert md['dtype'] == vs._dtype
        assert "n_classes" in md
        assert md['n_classes'] == vs._n_classes
        """
        for constant class shapers, name is assigned dynamically so that there are not duplicates if multiple 
        constants are used in the same batch generator. To make unique names, a class-based counter is used. 
        It is shared between all instances of the class
        """
        assert vs._dummy_constant_counter == 0
        vs1 = VarShaper(None, 1.)
        assert vs._dummy_constant_counter == 1
        assert vs1._dummy_constant_counter == 1
        assert vs1.metadata['name'] == 'dummy_constant_0'
        vs2 = VarShaper(None, 0.)
        assert vs._dummy_constant_counter == 2
        assert vs1._dummy_constant_counter == 2
        assert vs2._dummy_constant_counter == 2
        assert vs2.metadata['name'] == 'dummy_constant_1'

    @pytest.mark.dependency(name='test_decoded_dtype', depends=['test_init'])
    def test_decoded_dtype(self):
        """
        This checks that original datatype is captured by the shaper correctly
        """
        vs = VarShaper('label', self.le, sample=self.df)
        assert vs._decoded_dtype == self.df['label'].dtype
        # test that decoded_dtype can be learnt during transform if the encoder has shape property and
        # sample is not provided when created
        a = A()
        vs = VarShaper('label', a)
        assert vs._decoded_dtype is None
        tr = vs.transform(self.df)
        assert vs._decoded_dtype == self.df['label'].dtype

    @pytest.mark.dependency(name='test_inverse_transform_encoder', depends=['test_decoded_dtype'])
    def test_inverse_transform_encoder(self):
        """
        This tests that encoder based inverse transform works properly:
        - inverse transform adds new column to a dataframe
        - new column has the same datatype as a column in the original dataframe
        - encode-decode cycle restores data 100% correct
        """
        vs = VarShaper('label', self.le, sample=self.df)
        tr = vs.transform(self.df)
        df = pd.DataFrame()
        assert df.shape == (0, 0)
        vs.inverse_transform(df, tr)
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (self.df.shape[0], 1)
        assert 'label' in df
        assert df['label'].dtype == self.df['label'].dtype
        assert df['label'].equals(self.df['label'])
        # test onehot encoding
        vs = VarShaper('var1', self.lb, sample=self.df)
        tr = vs.transform(self.df)
        vs.inverse_transform(df, tr)
        assert df.shape == (self.df.shape[0], 2)
        assert 'var1' in df
        assert df['var1'].dtype == self.df['var1'].dtype
        assert df['var1'].equals(self.df['var1'])
        # test that datatypes are restored correctly
        le = LabelEncoder().fit([.0, 1, 2])
        tr = le.transform(self.df['var3'])
        iv = le.inverse_transform(tr)
        assert iv.dtype.kind == "f"
        vs = VarShaper('var3', le, sample=self.df)
        tr = vs.transform(self.df)
        vs.inverse_transform(df, tr)
        assert df.shape == (self.df.shape[0], 3)
        assert 'var3' in df
        assert df['var3'].dtype == self.df['var3'].dtype
        assert df['var3'].dtype.kind == 'i'
        assert df['var3'].equals(self.df['var3'])

    @pytest.mark.dependency(name='test_inverse_transform_const', depends=['test_inverse_transform_encoder'])
    def test_inverse_transform_const(self):
        """
        This test verifies that constant class is not inverse-transformed. (it is just ignored)
        """
        vs = VarShaper(None, 0.)
        tr = vs.transform(self.df)
        df = pd.DataFrame()
        assert df.shape == (0, 0)
        vs.inverse_transform(df, tr)
        # above inverse transform must not add any columns to df
        assert df.shape == (0, 0)

    @pytest.mark.dependency(name='test_inverse_transform_direct', depends=['test_inverse_transform_encoder'])
    def test_inverse_transform_direct(self):
        """
        This test verifies that direct class is inverse-transformed without losses
        """
        vs = VarShaper('var3', None)
        tr = vs.transform(self.df)
        df = pd.DataFrame()
        assert df.shape == (0, 0)
        vs.inverse_transform(df, tr)
        # above inverse transform must not add any columns to df
        assert df.shape == (self.df.shape[0], 1)
        assert df['var3'].dtype == self.df['var3'].dtype
        df['var3'].equals(self.df['var3'])
