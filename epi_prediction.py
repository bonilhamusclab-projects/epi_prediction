import math
import os

import nibabel as nib
from nilearn.image import mean_img
from nilearn.image import mean_img
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


def get_epi_paths(src_dir, pat_filter, con_filter):
    """
    Assumes src_dir has a folder for each of the modularities (dax, dmean, etc...)
    pat_filter and con_filter take file name as input
    """
    get_files = lambda mod, flr: [os.path.join(mod, f) for f in os.listdir(mod) if flr(f)]

    modularities = [(mod, os.path.join(src_dir, mod)) for mod in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, mod))]

    def add_modularity(acc, mod_p):
        modularity = mod_p[0]
        modularity_path = mod_p[1]
        ret = {}
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
    cons    [a0, a1]
    pats    [b0, b1]
    dtype: object
    >>> test
    cons    [a2]
    pats    [b2]
    dtype: object
    >>> paths = pd.Series({'cons': ['a0', 'a1', 'a2', 'a3'], 'pats': ['b0', 'b1', 'b2']})
    >>> (training, test) = split_training_test(paths, .67)
    >>> training
    cons    [a0, a1]
    pats    [b0, b1]
    dtype: object
    >>> test
    cons    [a2, a3]
    pats        [b2]
    dtype: object
    """
    ixs = {t: int(math.floor(train_ratio*len(series[t]))) for t in ['cons', 'pats']}
    training = lambda t: series[t][:ixs[t]]
    test = lambda t: series[t][ixs[t]:]
    return (pd.Series({'cons':training('cons'), 'pats': training('pats')}), \
            pd.Series({'cons': test('cons'), 'pats': test('pats')}))


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
                 a         b
    cons  [a0, a1]  [a0, a1]
    pats  [b0, b1]  [b0, b1]
    >>> test
             a         b
    cons  [a2]  [a2, a3]
    pats  [b2]      [b2]
    """
    training_df = pd.DataFrame()
    test_df = pd.DataFrame()
    for modularity in epi_paths.columns:
        (training, test) = split_training_test(epi_paths[modularity], train_ratio)
        training_df[modularity] = training
        test_df[modularity] = test
    return (training_df, test_df)


def load_data(series):
    load_path = lambda p: nib.load(p)

    return series.map(lambda paths: map(load_path, paths))


def masker(imgs):
	from nilearn.input_data import NiftiMasker
	ret = NiftiMasker(standardize=True)
	ret.fit(imgs)
	return ret


def data_to_2d(imgs, masker):
	return masker.transform(imgs)

def train(training_matrix, labels, k = 500):
	feature_selection = SelectKBest(f_classif, k = k)
	svc = SVC(kernel='linear')
	anova_svc = Pipeline([('anova', feature_selection), ('svc', svc)])
	anova_svc.fit(training_matrix, labels)
	return anova_svc


if __name__ == "__main__":
    import doctest
    doctest.testmod()
