from __future__ import division

import math
import os
import sys

from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import nibabel as nib
from nilearn import image
from nilearn.plotting import plot_glass_brain
import nilearn as nil
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import f1_score, mean_squared_error, precision_score, recall_score, make_scorer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


class NormalizerPipeline(BaseEstimator, TransformerMixin):
    def __init__(self, normalize_flag=True):
        self.normalize_flag = normalize_flag
        self.mu = None
        self.sigma = None
        self.w = None

    def fit(self, x, y):
        self.mu = np.mean(x, axis=0)
        self.sigma = np.std(x, axis=0)
#        self.w = np.apply_along_axis(self.vec_sum,axis=0,X)
        self.w = np.linalg.norm(x, axis=0)
        return self

    def transform(self, mat):
        if not self.normalize_flag:
            return mat

        return ((mat - self.mu) / self.sigma) / self.w

    @staticmethod
    def vec_sum(array):
        return np.sqrt(np.sum(np.square(array)))

    
class SimpleMaskerPipeline(BaseEstimator, TransformerMixin):
    def __init__(self, threshold):
        self.threshold = threshold
        self.mask_image = nib.load('masks/white.nii')
    
    def fit(self, x, y):
        return self
    
    def transform(self, mat):
        self.indexes = self.mask_image.get_data().flatten() >= self.threshold
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
    

def sensitivity(expected, predicted):
    """
    >>> import numpy as np
    >>> expected = np.array([1, 1, 0, 0])
    >>> predicted = np.array([1, 0, 1, 1])
    >>> assert(sensitivity(expected, predicted) == .5)
    >>> predicted = np.array([1, 1, 0, 1])
    >>> assert(sensitivity(expected, predicted) == 1)
    >>> predicted = np.array([0, 0, 1, 0])
    >>> assert(sensitivity(expected, predicted) == 0)    
    """
    all_pos = np.sum(expected)
    true_pos_found = np.sum(predicted[expected > 0])
    return true_pos_found/all_pos


def specificity(expected, predicted):
    """
    >>> import numpy as np
    >>> expected = np.array([1, 0, 1, 0])
    >>> predicted = np.array([1, 1, 0, 0])
    >>> res = specificity(expected, predicted)
    >>> assert(res == .5)
    >>> predicted = np.array([1, 0, 0, 0])
    >>> assert(specificity(expected, predicted) == 1)
    >>> predicted = np.array([0, 1, 0, 1])
    >>> assert(specificity(expected, predicted) == 0)    
    """
    neg_ixs = expected == 0
    all_neg = np.sum(neg_ixs)
    true_neg_found = np.sum(predicted[neg_ixs] == 0)
    return true_neg_found/all_neg


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
    
    def sensitivity_scores(self):
        return self._compare(sensitivity)
    
    def avg_sensitivity_score(self):
        return np.average(self.sensitivity_scores())
    
    def specificity_scores(self):
        return self._compare(specificity)
    
    def avg_specificity_score(self):
        return np.average(self.specificity_scores())    

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


def load_data(series):
    load_path = lambda p: nib.load(p)

    return series.map(lambda paths: map(load_path, paths))


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


def load_mat_and_labels(src_dir, mod):
    control_filter = lambda file_name: 'CON' in file_name
    patient_filter = lambda file_name: 'PAT' in file_name
    epi_paths = get_epi_paths(src_dir, patient_filter, control_filter)
    mod_paths = epi_paths[mod]
    labels = len(mod_paths['pats']) * [1] + len(mod_paths['cons']) * [0]
    labels_arr = np.array(labels)

    wm_image = nib.load('masks/white.nii')

    def verbose_load(f):
        sys.stdout.write('#')
        return nil.image.resample_img(f,
                                      target_shape=wm_image.shape,
                                      target_affine=wm_image.get_affine())

    fs = mod_paths['pats'] + mod_paths['cons']
    mat = np.vstack([verbose_load(f).get_data().flatten() for f in fs])
    return mat, labels_arr


def run(src_dir, mod):

    if isinstance(src_dir, str):
        mat, labels_arr = load_mat_and_labels(src_dir, mod)
    else:
        mat, labels_arr = (src_dir, mod)

    masker = SimpleMaskerPipeline(threshold=.2)
    normalizer = NormalizerPipeline()
    svc = SVC(kernel='linear')

    pipeline = Pipeline( [('masker', masker),
                          ('normalizer', normalizer),
                          ('anova', SelectKBest(k=500)),
                          ('svc', svc)] )

    c_range = [.01, .1, 1, 10, 100]
    t_range = [.2, .5, .7, .9]
    k_range = [50, 250, 500]
    flag_range = [True, False]

    param_grid = {"svc__C": c_range,
                  "masker__threshold": t_range,
                  "anova__k": k_range,
                  "normalizer__normalize_flag": flag_range
                  }

    cv = StratifiedKFold(labels_arr, n_folds=6)

    total_runs = len(c_range) * 2 * len(t_range) *2 * len(k_range) * 2 * len(flag_range) * 2
    scorer = verbose_scorer(total_runs)

    grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=cv, scoring = scorer)
    grid_search.fit(mat, labels_arr)

    return grid_search


def run_ensemble(src_dir, dmean_params, kmean_params, fa_params):
    params_map = dict(dmean=dmean_params, kmean=kmean_params, fa=fa_params)

    load_mat = lambda mod: load_mat_and_labels(src_dir, mod)[0]
    mat = np.hstack([load_mat('dmean'), load_mat('fa'), load_mat('kmean')])
    labels = load_mat_and_labels(src_dir, 'dmean')[1]

    dist = int(mat.shape[1]/3)
    index_map = {"dmean": (0, dist),
                 "fa":  (dist, 2 * dist),
                 "kmean": (2*dist, 3*dist)}

    def new_pipe(mod):
        svc = SVC()
        svc.kernel='linear'
        svc.C = params_map[mod]['C']
        svc.probability=True
        k = params_map[mod]['k']
        thresh = params_map[mod]['thresh']
        masker = SimpleMaskerPipeline(thresh)
        normalize_flag = params_map[mod]['normalize_flag']
        normalizer = NormalizerPipeline(normalize_flag=normalize_flag)
        return Pipeline([
            ('columns', ColumnSelector(index_map[mod])),
            ('whitematter', masker),
            ('normalizer', normalizer),
            ('anova', SelectKBest(k=k)),
            ('svc', svc)
        ])

    algs = [new_pipe(m) for m in params_map]
    combined_alg = ProbableBinaryEnsembleAlg(algs)
    cv_combos, cv_combos_train = verbose_cv(mat, labels, combined_alg, n_folds=6, verbose=False)

    no_fa_alg = ProbableBinaryEnsembleAlg([new_pipe(m) for m in ['kmean', 'dmean']])
    cv_combos_no_fa, cv_combos_train_no_fa = verbose_cv(mat, labels, no_fa_alg, n_folds=6, verbose=False)

    no_kmean_alg = ProbableBinaryEnsembleAlg([new_pipe(m) for m in ['fa', 'dmean']])
    cv_combos_no_kmean, cv_combos_train_no_kmean = verbose_cv(mat, labels, no_kmean_alg, n_folds=6, verbose=False)

    no_dmean_alg = ProbableBinaryEnsembleAlg([new_pipe(m) for m in ['fa', 'kmean']])
    cv_combos_no_dmean, cv_combos_train_no_dmean = verbose_cv(mat, labels, no_dmean_alg, n_folds=6, verbose=False)

    ret = dict(all=(cv_combos, cv_combos_train),
               no_fa=(cv_combos_no_fa, cv_combos_train_no_fa),
               no_dmean=(cv_combos_no_dmean, cv_combos_train_no_dmean),
               no_kmean=(cv_combos_no_kmean, cv_combos_train_no_kmean))

    return ret


def calc_coeffs(cv, fit_fn, coeffs_fn, normalize=True):
    coeffs = None
    for train, test in cv:
        fit_fn(train)
        coeffs_step = coeffs_fn()
        if normalize:
            coeffs_step = coeffs_step/np.sum(np.abs(coeffs_step))
        coeffs = coeffs_step if coeffs is None else coeffs + coeffs_step

    return coeffs/len(cv)


def plot_bw_coeffs(coeffs, affine, title, base_brightness=.7, cmap=None, output_file=None, black_bg=False):
    def isstr(s): return isinstance(s, str)

    def default_cmap():
        invert_if_black_bg = lambda v: (1 - v) if black_bg else v
        base_brightness_local = invert_if_black_bg(base_brightness)
        end_brightness = invert_if_black_bg(0)
        avg = np.average([base_brightness_local, end_brightness])
        c_range = ((0, base_brightness_local, base_brightness_local),
                   (.33, base_brightness_local, avg),
                   (.67, avg, end_brightness),
                   (1, end_brightness, end_brightness))
        c_dict = {r: c_range for r in ['red', 'green', 'blue']}
        cmap_name = 'bright_bw'
        cmap = LinearSegmentedColormap(cmap_name, c_dict)
        plt.register_cmap(cmap=cmap)
        return cmap

    cmap = plt.get_cmap(cmap) if isstr(cmap) else default_cmap() if cmap is None else cmap

    plot_glass_brain(nib.Nifti1Image(coeffs, affine=affine),
                     title=title,
                     black_bg=black_bg,
                     colorbar=True,
                     output_file=output_file,
                     cmap=cmap,
                     alpha=.15)


def plot_coeffs(coeffs, affine, neg_disp=.8, pos_disp=.8, **kwargs):

    def default_cmap():
        max_neg_coeff = np.abs(np.min(coeffs))
        max_pos_coeff = np.max(coeffs)
        max_coeff = np.max((max_neg_coeff, max_pos_coeff))

        dev = 0.5

        neg_dev = dev * max_neg_coeff/max_coeff
        pos_dev = dev * max_pos_coeff/max_coeff

        zero = 0.5
        max_neg = zero - neg_dev
        max_pos = zero + pos_dev

        na_clr = .5
        na_start = (0.0, na_clr, na_clr)
        na_end = (1.0, na_clr, na_clr)

        #pos_80 = pos_dev * np.percentile(coeffs[coeffs > 0], pos_disp*100)/max_pos_coeff
        #neg_80 = neg_dev * np.percentile(np.abs(coeffs[coeffs < 0]), neg_disp*100)/max_neg_coeff

        blue_red_bp_map = {
            'red': (
                na_start,
                (max_neg, na_clr, 0.0),
                (zero, 0.0, 1.0),
                (max_pos, 1.0, na_clr),
                na_end
            ),
            'blue': (
                na_start,
                (max_neg, na_clr, 0.0),
                (zero - neg_disp*neg_dev, 1.0, 1.0),
                (zero, 1.0, 0.0),
                (max_pos, 0.0, na_clr),
                na_end
            ),
            'green': (
                na_start,
                (max_neg, na_clr, 1.0),
                #(zero - neg_80, 1-neg_disp, 1-neg_disp),
                (zero - neg_disp*neg_dev, 1.0, 1.0),
                (zero, 0.0, 0.0),
                #(zero + pos_80, 1-pos_disp, 1-pos_disp),
                (zero + pos_disp*pos_dev, pos_disp, pos_disp),
                (max_pos, 1.0, na_clr),
                na_end
            )
            }

        name = 'BlueRedBiPolar'
        return LinearSegmentedColormap(name, blue_red_bp_map)

    img = nib.Nifti1Image(coeffs, affine=affine)
    kwargs['cmap'] = default_cmap()
    plot_glass_brain(img, **kwargs)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
