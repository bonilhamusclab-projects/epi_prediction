from __future__ import division

import math
import os
import sys

import matplotlib.pyplot as plt
import nibabel as nib
from nilearn import image
import numpy as np
import pandas as pd
from scipy.stats.mstats import zscore
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import f1_score, mean_squared_error, precision_score, recall_score, make_scorer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

class NormalizerPipeline(BaseEstimator, TransformerMixin):
    def __init__(self, normalize_flag=True):
        self.normalize_flag = normalize_flag
        
    def fit(self, X, y):
        ##calculate, min, max, std etc.. from training data
        return self
    
    def transform(self, mat):
        if not self.normalize_flag:
            return mat
        
        ##perform calculations here
        ret = mat
        
        return ret

    
class SimpleMaskerPipeline(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0):
        self.threshold = threshold
        self.mask_image = nib.load('masks/white.nii')
        self.indexes = self.mask_image.get_data().flatten() >= threshold
    
    def fit(self, X, y):
        return self
    
    def transform(self, mat):
        return mat[:, self.indexes]
            


class SimpleMasker:
    def __init__(self, mask_image, threshold=None):
        self.mask_image = mask_image
        self._mask_image = nib.load(mask_image)
        self._threshold = threshold
        if self._threshold is not None:
            data = self._mask_image.get_data()
            data[data < threshold] = 0
            self._mask_image = nib.Nifti1Image(data, self._mask_image.get_affine())
        self._indexes = self._mask_image.get_data().nonzero()
        
    def update_threshold(self, threshold):
        self._threshold = threshold
        data = self._mask_image.get_data()
        data[data < threshold] = 0
        self._mask_image = nib.Nifti1Image(data, self._mask_image.get_affine())
        self._indexes = self._mask_image.get_data().nonzero()

    def transform(self, f):
        if isinstance(f, str):
            f = nib.load(f)
        if f.shape != self._mask_image.shape:
            f = image.resample_img(f, 
                    target_shape=self._mask_image.shape,
                    target_affine=self._mask_image.get_affine())
        return np.array(f.get_data()[self._indexes])

    def transform_many(self, fs, verbose=False):
        num_fs = len(fs)

        if verbose:
            print('-' * num_fs)

        def print_if_verbose(f_name):
            if verbose:
                sys.stdout.write('#')
            return self.transform(f_name)

        return np.vstack([print_if_verbose(f) for f in fs])

    def inv_transform(self, arr, affine=None):
        shape = self._mask_image.shape
        data = np.zeros(shape)
        data[self._indexes] = arr
        return nib.Nifti1Image(data, affine) if affine is not None else data


class CvInfo:
    def __init__(self, expected_mat, predicted_mat):
        self.expected_mat = expected_mat
        self.predicted_mat = predicted_mat
        self.expected_predicted_paired = zip(expected_mat, predicted_mat)

    def _compare(self, fn):
        return [fn(expected, predicted)
                for expected, predicted in self.expected_predicted_paired]

    def avg_f1_score(self):
        return np.average(self.f1_scores())

    def f1_scores(self):
        return self._compare(f1_score)

    def root_mean_square_errors(self):
        return self._compare(lambda e, p: math.sqrt(mean_squared_error(e, p)))

    def avg_rmse(self):
        return np.average(self.root_mean_square_errors())

    def precision_scores(self):
        return self._compare(precision_score)

    def avg_precision(self):
        return np.average(self.precision_scores())

    def recall_scores(self):
        return self._compare(recall_score)

    def avg_recall(self):
        return np.average(self.recall_scores())
    
    def confusion_matrix(self):
        from sklearn.metrics import confusion_matrix
        ret = None
        cms = self._compare(confusion_matrix)
        for cm in cms:
            ret = cm if ret is None else ret + cm
        return ret
    
    def normalized_confusion_matrix(self):
        cm = self.confusion_matrix()
        return cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    def __repr__(self):
        new_line = lambda s, v: s.format(v) + "\n"

        return (new_line("f1 scores: {}", self.f1_scores()) +
                new_line("avg f1 score: {}", self.avg_f1_score()) +
                new_line("precision scores: {}", self.precision_scores()) +
                new_line("avg precision score: {}", self.avg_precision()) +
                new_line("recall scores: {}", self.recall_scores()) +
                new_line("avg recall score: {}", self.avg_recall()) +
                new_line("root-mean-square-error scores: {}", self.root_mean_square_errors()) +
                new_line("avg root-mean-square-error: {}", self.avg_rmse()))


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
    ixs = {t: int(math.floor(train_ratio * len(series[t]))) for t in ['cons', 'pats']}

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


def hstack(simple_masker, verbose, *fs_for_modality):
    return np.hstack([simple_masker.transform_many(fs, verbose)
                      for fs in
                      fs_for_modality])


def verbose_cv(mat, labels, alg, n_folds=3, verbose=True):
    cv = StratifiedKFold(labels, n_folds=n_folds)

    expected_mat = []
    predicted_mat = []

    expected_mat_train = []
    predicted_mat_train = []

    def print_v(s):
        if verbose:
            print s

    for train, test in cv:
        expected = labels[test]
        print_v("train labels")
        print_v(labels[train])
        print_v("test labels")
        print_v(expected)
        print_v("about to fit")
        alg.fit(mat[train], labels[train])
        print_v("about to predict")
        predicted = alg.predict(mat[test])
        print('####')

        expected_mat.append(expected)
        predicted_mat.append(predicted)

        expected_mat_train.append(labels[train])
        predicted_mat_train.append(alg.predict(mat[train]))

    return CvInfo(expected_mat, predicted_mat), CvInfo(expected_mat_train, predicted_mat_train)


def verbose_scorer(total_runs, score_fn=f1_score):
    print('-' * total_runs)

    def verbose_score_fn(truth, predictions):
        sys.stdout.write('#')
        return score_fn(truth, predictions)

    return make_scorer(verbose_score_fn)


def plot_confusion_matrix(cm, title, cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, [0, 1])
    plt.yticks(tick_marks, [0, 1])
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, col_indexes):
        self.col_indexes = col_indexes

    def fit(self, x, y=None):
        return self

    def transform(self, mat):
        start = self.col_indexes[0]
        stop = self.col_indexes[1]
        return mat[:, start:stop]
    

class RowCombiner(BaseEstimator, TransformerMixin):
    def __init__(self, num_cols):
        self.num_cols = num_cols
        
    def fit(self, x, y=None):
        return self
    
    def transform(self, labels):
        num_cols = self.num_cols
        ret = np.zeros(len(labels)/num_cols, num_cols)
        for ix in range(0, len(labels), num_cols):
            ret[ix] = labels[ix:(ix+num_cols)]
        return ret


class ProbableBinaryEnsembleAlg:
    def __init__(self, algs):
        self.algs = algs

    def fit(self, X, y):
        for alg in self.algs:
            alg.fit(X, y)
        return self

    def predict(self, X):
        probs = None
        for alg in self.algs:
            alg_probs = alg.predict_proba(X)
            probs = alg_probs if probs is None else probs + alg_probs
        predictions = np.zeros((len(probs), 1))
        predictions[probs[:, 0] < probs[:, 1]] = 1
        return predictions


if __name__ == "__main__":
    import doctest

    doctest.testmod()
