import math
import os

import nibabel as nib
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


class SimpleMasker:
    def __init__(self, mask_image):
        self.mask_image = mask_image
        self._mask_image = nib.load(mask_image)
        self._indexes = self._mask_image.get_data().nonzero()

    def transform(self, f):
        if isinstance(f, str):
            f = nib.load(f)
        return np.array(f.get_data()[self._indexes])

    def inv_transform(self, arr, affine=None):
        shape = self._mask_image.shape
        data = np.zeros(shape)
        data[self._indexes] = arr
        return nib.Nifti1Image(data, affine) if affine is not None else data


def get_epi_paths(src_dir, pat_filter, con_filter):
    """
    Assumes src_dir has a folder for each of the modularities (dax, dmean, etc...)
    pat_filter and con_filter take file name as input
    """
    get_files = lambda mod, flr: [os.path.join(mod, f) for f in os.listdir(mod) if flr(f)]

    modularities = [(mod, os.path.join(src_dir, mod))
                    for mod in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, mod))]

    def add_modularity(acc, mod_p):
        modularity = mod_p[0]
        modularity_path = mod_p[1]
        ret = dict()
        ret['pats'] = get_files(modularity_path, pat_filter)
        ret['cons'] = get_files(modularity_path, con_filter)
        acc[modularity] = ret
        return acc
        
    return pd.DataFrame(reduce(add_modularity, modularities, dict()))


def split_training_test(series, train_ratio):
    """
    >>> import pandas as pd
    >>> paths = pd.Series({'cons': ['a0', 'a1', 'a2'], 'pats': ['b0', 'b1', 'b2']})
    >>> (training, test) = split_training_test(paths, .67)
    >>> training
    data      [a0, a1, b0, b1]
    labels        [0, 0, 1, 1]
    dtype: object
    >>> test
    data      [a2, b2]
    labels      [0, 1]
    dtype: object
    >>> paths = pd.Series({'cons': ['a0', 'a1', 'a2', 'a3'], 'pats': ['b0', 'b1', 'b2']})
    >>> (training, test) = split_training_test(paths, .67)
    >>> training
    data      [a0, a1, b0, b1]
    labels        [0, 0, 1, 1]
    dtype: object
    >>> test
    data      [a2, a3, b2]
    labels       [0, 0, 1]
    dtype: object
    """
    ixs = {t: int(math.floor(train_ratio*len(series[t]))) for t in ['cons', 'pats']}

    training = series['cons'][:ixs['cons']] + series['pats'][:ixs['pats']]
    training_labels = [0] * ixs['cons'] + [1] * ixs['pats']

    test = series['cons'][ixs['cons']:] + series['pats'][ixs['pats']:]
    test_labels = [0] * (len(series['cons']) - ixs['cons']) + [1] * (len(series['pats']) - ixs['pats'])

    return (pd.Series({'data': training, 'labels': training_labels}),
            pd.Series({'data': test, 'labels': test_labels}))


def split_training_tests(epi_paths, train_ratio):
    """
    >>> import pandas as pd
    >>> paths_a = pd.Series({'cons': ['a0', 'a1', 'a2'], 'pats': ['b0', 'b1', 'b2']})
    >>> paths_b = pd.Series({'cons': ['a0', 'a1', 'a2', 'a3'], 'pats': ['b0', 'b1', 'b2']})
    >>> df = pd.DataFrame({'a': paths_a, 'b': paths_b})
    >>> (training, test) = split_training_tests(df, .7)
    >>> training.columns
    Index([u'a', u'b'], dtype='object')
    >>> test.columns
    Index([u'a', u'b'], dtype='object')
    >>> training
                           a                 b
    data    [a0, a1, b0, b1]  [a0, a1, b0, b1]
    labels      [0, 0, 1, 1]      [0, 0, 1, 1]
    >>> test
                   a             b
    data    [a2, b2]  [a2, a3, b2]
    labels    [0, 1]     [0, 0, 1]
    """
    training_df = pd.DataFrame()
    test_df = pd.DataFrame()
    for modularity in epi_paths.columns:
        (training, test) = split_training_test(epi_paths[modularity], train_ratio)
        training_df[modularity] = training
        test_df[modularity] = test
    return training_df, test_df


def load_data(series):
    load_path = lambda p: nib.load(p)

    return series.map(lambda paths: map(load_path, paths))


def gen_masker(images):
    from nilearn.input_data import NiftiMasker
    ret = NiftiMasker(standardize=True)
    ret.fit(images)
    return ret


def data_to_2d(images, masker):
    return masker.transform(images)


def train(training_matrix, labels, k=500):
    feature_selection = SelectKBest(f_classif, k=k)
    svc = SVC(kernel='linear')
    anova_svc = Pipeline([('anova', feature_selection), ('svc', svc)])
    anova_svc.fit(training_matrix, labels)
    return anova_svc


if __name__ == "__main__":
    import doctest
    doctest.testmod()
