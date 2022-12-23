import numpy as np
from scipy.interpolate import interp1d
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
import KDEpy    

import dill as pickle

def fit_sklearn_pair_kde(data):
    params = {'bandwidth': np.logspace(-3, 3, 50)}

    grid = GridSearchCV(KernelDensity(kernel='gaussian', rtol=1e-4), params)
    grid.fit(data.reshape(-1, 1))
    kde = grid.best_estimator_
#     print(kde.get_params())
    return kde

def fit_fast_kde_interpolator(data, weights=None):
    # TODO - weights parameter not set 
    kdepy_model = KDEpy.FFTKDE(kernel='gaussian', bw='ISJ').fit(data, weights)
    kdepy_x, kdepy_y = kdepy_model.evaluate()       

    
    # use interpolated spline to speed up future scoring
    spline_approximation = interp1d(kdepy_x, np.log(kdepy_y), fill_value='extrapolate')
    return spline_approximation

def fit_fresco_on_pcore_histograms(dict_of_frag_ensemble_histograms,
                                   pcore_pairs,
                                   dict_of_frag_ensemble_weights=None):

    fresco_kdes = {}
    for pcore_pair in pcore_pairs:
        fresco_kdes[pcore_pair] = fit_fast_kde_interpolator(dict_of_frag_ensemble_histograms[pcore_pair],
                                                            weights=dict_of_frag_ensemble_weights[pcore_pair])
    return fresco_kdes

def score_dist(kde, dist):
    if len(dist):  # non-zero length

        # log-prob (larger = higher prob)
        score = kde(dist.reshape(-1, 1))
        score = np.max(score)

        return score
    else:
        return np.nan
    
def load_kde_model(filename):
    with open(filename, 'rb') as f:
        kde_model = pickle.load(f)
    return kde_model
    
def save_kde_model(kde_model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(kde_model, f)
    