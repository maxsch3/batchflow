import numpy as np
from .batch_transformer import BatchTransformer


class BaseRandomCellTransform(BatchTransformer):

    """ This is a base class used for all sorts of random cell transforms: feature dropout, noise, etc

    The transform is working by masked replacement of cells in a batch with some augmented version of the same batch:

    batch.loc[mask] = augmented_batch[mask]

    The this transform will provide infrastructure of this transformation, while derived classes will
    define their own versions of augmented batch

    **Parameters:**
    - **n_probs** - a *list*, *tuple* or a *one dimensional numpy array* of probabilities $p_0, p_1, p_2, ... p_n$.
        $p_0$ is a probability for a row to have 0 augmented elements (no augmentation), $p_1$ - one random cells,
        $p_2$  - two random cells, etc. A parameter must have at least 2 values, scalars are not accepted
    - **cols* - a *list*, *tuple* or *one dimensional numpy array* of strings with columns names to be transformed
        Number of columns must be greater or equal the length of **n_probs** parameter simply because there must be
        enough columns to choose from to augment n elements in a row
    - **col_probs** - (optional) a *list*, *tuple* or *one dimensional numpy array* of floats
        $p_{c_0}, p_{c_1}, ... p_{c_k}$ where k is the number of columns specified in parameter cols.
        $p_{c_0}$ is the probability of column 0 from parameter *cols* to be selected in when only one column is picked
        for augmentation. $p_{c_1}$ is the same for column 1, etc. It is important to understand that when two or more
        columns, are picked for a row, actual frequencies of columns will drift towards equal distribution with every
        new item added. In a case when number of columns picked for augmentation reaches its max allowed value
        (number of columns available to choose from parameter **cols**), there will be no choice and the actual counts
        of columns will be equal. This means the actual distribution will turn into a uniform discrete distribution.
        **Default: None**

    """

    def __init__(self, n_probs, cols, col_probs=None):
        super().__init__()
        # if type(row_prob) not in [float]:
        #     raise ValueError('Error: row_prob must be a scalar float')
        # self._row_prob = row_prob

        # checking cols and col_probs
        self._cols = self._validate_cols(cols)
        self._col_probs = self._validate_col_probs(col_probs)
        # checking n_probs
        self._n_probs = self._validate_n_probs(n_probs)
        self._col_factors = self._calculate_col_weights(self._col_probs)

        # seed table is a technical object used for vectorised implementation of n_probs
        self._seed = np.tril(np.ones((self._n_probs.shape[0], self._n_probs.shape[0]-1), dtype=np.int64), k=-1)

    def _validate_vector(self, vector, name, numeric=True):
        if vector is None:
            return None
        if type(vector) not in [list, tuple, int, float, np.ndarray]:
            raise ValueError(f'Error. Parameter {name} must be one of the following types: list, tuple, single int or'
                             f' float. Got {type(vector)}')
        if type(vector) in [int, float]:
            vector = [vector]
        if numeric:
            try:
                v = np.array(vector, dtype=float)
            except ValueError:
                raise ValueError(f'Error. parameter {name} must be all-numeric')
        if type(vector) != np.ndarray:
            vector = np.array(vector)
        if vector.ndim > 1:
            raise ValueError(f'Error. parameter {name} must be a one-dimensional array or a scalar')
        return vector

    def _validate_n_probs(self, n_probs):
        if n_probs is None:
            raise ValueError(f'Error. parameter n_probs must be set')
        nb = self._validate_vector(n_probs, 'n_probs')
        if nb.shape[0] > self._cols.shape[0] + 1:
            raise ValueError(f'Error. Length of parameter n_probs must not be lower that the length of parameter'
                             f' cols + 1. There must be enough columns to choose from to fill all positions specified'
                             f' in n_probs')
        if np.abs(nb.sum() - 1.) > 0.001:
            raise ValueError('Error. n_probs do not add up to 1.')
        if nb.shape[0] < 2:
            raise ValueError('Error. Parameter n_probs must have at least 2 values')
        return nb

    def _validate_cols(self, cols):
        if type(cols) == str:
            cols = [cols]
        if type(cols) in [list, tuple, np.ndarray]:
            if not all([type(s) == str for s in cols]):
                raise ValueError('Error: parameter cols can only contain strings')
        else:
            raise ValueError('Error: parameter cols must be a single column name a list of column names')
        c = np.array(cols)
        return c

    def _validate_col_probs(self, col_probs):
        if col_probs is None:
            cp = np.ones(shape=self._cols.shape)
        else:
            cp = self._validate_vector(col_probs, 'col_probs')
            if len(cp) == 1:
                raise ValueError('Error. parameter col_probs is not accepted when only one column in augmented')
        if cp.shape != self._cols.shape:
            raise ValueError('Error. parameters cols and col_probs must have same shape')
        return cp

    def _make_mask(self, batch):
        """ This method creates a binary mask that marks items in an incoming batch that have to be augmented

        The elements are selected taking the following parameters into account:

        - n_probs - list of probabilities of picking 0, 1,... respectively
        - cols - list of columns subjected to
        - col_probs -


        **Parameters:**

        **Returns:** a pandas dataframe of booleans of the same dimensions and indices as a batch. The returned
        dataframe has True for elements that have to be augmented

        """
        rand = np.random.uniform(size=(batch.shape[0], self._cols.shape[0]))
        # idx is a rectangular matrix of ids of columns selected randomly with column weighting
        idx = np.argsort(np.power(rand, self._col_factors))[:, :(self._n_probs.shape[0]-1)] + 1
        # now I will create a mask implementing n_probs (randomly picking rows with 0, 1, 2 etc cells picked)
        seed_idx = np.random.choice(range(self._seed.shape[0]), size=idx.shape[0], p=self._n_probs)
        idx = idx * self._seed[seed_idx, :]
        b = np.zeros((idx.shape[0], self._cols.shape[0] + 1))
        for i in range(idx.shape[1]):
            b[np.arange(idx.shape[0]), idx[:, i]] = 1
        return b[:, 1:]

    def _calculate_col_weights(self, col_probs):
        """ Calculate power factors for transformation according to desired frequencies

        The weighed col sampler is using vectorized argsort for selecting unqiue ids in each row.
        The downside of this approach is that it doesn't use weighting.
        I.e. I can't make one column more prefferable if there is a choice of columns in each row.
        When using uniform distribution as is, all variables become equally possible which means each
        column can be selected with equal probability when only one column is chosen

        to illustrate why this is happening, I will use CDF of a uniform distribution $X$
        For a standard uniform distribution in unit interval $[0,1]$, the CDF fuction is

        $$
        CDF(X) = x : 0\le x\le 1
        $$

        CDF sets the probability of a random variable to evaluate less than x

        $$
        CDF(X, x) = p(X \le x)
        $$

        I can calculate probability of one variable be less than another $p(X_1 \le X_2)$.
        For that I need to integrate the CDF:

        $$
        p(X_1 \le X_2) = \int_0^1 CDF(X_2) dX_1 = \int_0^1 x dX_1 = \int_0^1 x \cdot 1 \cdot dx =
        \bigl(\frac{x^2}{2} + C\bigr) \biggr\rvert_0^1 = \frac{1}{2}
        $$

        then 3 variables are used, I will calculate joint probability

        $$
        p(X_1 \le X_2, X_1 \le X_3) = \int_0^1 CDF(X_2) \cdot CDF(X_3) \cdot dX_1 =
        \int_0^1 x^2 dX_1 = \int_0^1 x^2 \cdot 1 \cdot dx =
        \bigl(\frac{x^3}{3} + C\bigr) \biggr\rvert_0^1 = \frac{1}{3}
        $$

        ## Adding weighting

        Now, how I can skew the outcomes, so that the expectations of them being chosen are not equal,
        but some other ratios? For that, I need to modify distribution of $X$ so that integral changes
        in the way we want. The distributions must not change their intervals and must stay within unit limits $[0,1]$.
        For this reason, I will not use simple scaling.
        Instead, I will use power transformation of standard uniform distribution.

        $$ X_1 = X^{z_1} $$
        $$ X_2 = X^{z_2} $$
        $$ X_3 = X^{z_3} $$

        The power factors $z_1, z_2, z_3$ are not yet known.
        Lets see if we can find them using desired weights $[p_1, p_2, p_3]$ for these variables:

        $$
        p_1 = p(X_1 \le X_2, X_1 \le X_3) = \int_0^1 CDF(X_2) \cdot CDF(X_3) \cdot dX_1 =
        \int_0^1 x^{z_2} x^{z_3} dX_1 =
        $$
        $$
        \int_0^1 x^{z_2} x^{z_3} \frac{dX_1}{dx} dx = \int_0^1 x^{z_2} x^{z_3} (z_1\cdot x^{z_1-1}) dx =
        $$
        $$
        z_1 \int_0^1 x^{z_2} x^{z_3} x^{z_1-1} dx = z_1 \int_0^1 x^{z_1+z_2+z_3-1} dx =
        $$
        $$
        z_1 \int_0^1 x^{z_1+z_2+z_3-1} dx =
        z_1 \bigl( \frac{x^{z_1+z_2+z_3-1}}{z_1+z_2+z_3-1} + C\bigr) \biggr\rvert_0^1 =
        \frac{z_1}{z_1+z_2+z_3-1}
        $$

        This makes a system of equations

        $$ p_1 = \frac{z_1}{z_1+z_2+z_3-1} $$
        $$ p_2 = \frac{z_2}{z_1+z_2+z_3-1} $$
        $$ p_3 = \frac{z_3}{z_1+z_2+z_3-1} $$

        or

        $$ (p_1 - 1) z_1+p_1z_2+p_1z_3 = 0 $$
        $$ p_2z_1+(p_2-1)z_2+p_2z_3 = 0 $$
        $$ p_3z_1+p_3z_2+(p_3-1)z_3 = 0 $$

        This unfortunately is a homogenious system of equations which has a simple solution:
        $Z = 0$ also the matrix is singular. This means that there are either none or multiple solutions.

        I will use SVD for finding one of the solution

        $$
        A Z = 0
        $$
        Matrix A can be decomposed to
        $$
        A = UDV^T
        $$

        The solution will be in the n-th column where zero diagonal element is in matrix $D$.
        For above matrix, this element will be on last position. The solution will be located in the last row of matrix V

        :return:
        """

        # first, normalize p
        cp = col_probs / col_probs.sum()
        # then create a matrix A
        a = np.tile(np.expand_dims(cp, -1), (1, cp.shape[0])) - np.eye(cp.shape[0])
        u, d, v = np.linalg.svd(a)
        weights = v[-1, :]
        # a is singular and might have multiple solutions: z=0, z, and -z. We only want positive z
        if weights.sum() < 0:
            weights *= -1
        return weights

    def _make_augmented_version(self, batch):
        raise NotImplemented()
        return batch

    def transform(self, batch):
        subset = batch[self._cols]
        mask = self._make_mask(subset)
        augmented_batch = self._make_augmented_version(subset)
        # batch.iloc[mask] = augmented_batch.iloc[mask]
        subset1 = subset.mask(mask > .5, augmented_batch.values)
        batch[self._cols] = subset1
        return batch
