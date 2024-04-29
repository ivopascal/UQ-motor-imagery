"""Module for classification function."""
"taken from: https://github.com/pyRiemann/pyRiemann/blob/master/pyriemann/classification.py and added uncertainty"

import functools

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.extmath import softmax
from sklearn.pipeline import make_pipeline
from joblib import Parallel, delayed
import warnings

from .utils.mean import mean_covariance
from .utils.distance import distance
from .utils.utils import check_metric


def _mode_1d(X):
    vals, counts = np.unique(X, return_counts=True)
    mode = vals[counts.argmax()]
    return mode


def _mode_2d(X, axis=1):
    mode = np.apply_along_axis(_mode_1d, axis, X)
    return mode

class MDM(BaseEstimator, ClassifierMixin, TransformerMixin):
    r"""Classification by Minimum Distance to Mean.

    For each of the given classes :math:`k = 1, \ldots, K`, a centroid
    :math:`\mathbf{M}^k` is estimated according to the chosen metric.

    Then, for each new matrix :math:`\mathbf{X}`, the class is affected
    according to the nearest centroid [1]_:

    .. math::
        \hat{k} = \arg \min_{k} d (\mathbf{X}, \mathbf{M}^k)

    Parameters
    ----------
    metric : string | dict, default="riemann"
        Metric used for mean estimation (for the list of supported metrics,
        see :func:`pyriemann.utils.mean.mean_covariance`) and
        for distance estimation
        (see :func:`pyriemann.utils.distance.distance`).
        The metric can be a dict with two keys, "mean" and "distance"
        in order to pass different metrics.
        Typical usecase is to pass "logeuclid" metric for the "mean" in order
        to boost the computional speed, and "riemann" for the "distance" in
        order to keep the good sensitivity for the classification.
    n_jobs : int, default=1
        The number of jobs to use for the computation. This works by computing
        each of the class centroid in parallel.
        If -1 all CPUs are used. If 1 is given, no parallel computing code is
        used at all, which is useful for debugging. For n_jobs below -1,
        (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one
        are used.

    Attributes
    ----------
    classes_ : ndarray, shape (n_classes,)
        Labels for each class.
    covmeans_ : ndarray, shape (n_classes, n_channels, n_channels)
        Centroids for each class.

    See Also
    --------
    Kmeans
    FgMDM
    KNearestNeighbor

    References
    ----------
    .. [1] `Multiclass Brain-Computer Interface Classification by Riemannian
        Geometry
        <https://hal.archives-ouvertes.fr/hal-00681328>`_
        A. Barachant, S. Bonnet, M. Congedo, and C. Jutten. IEEE Transactions
        on Biomedical Engineering, vol. 59, no. 4, p. 920-928, 2012.
    .. [2] `Riemannian geometry applied to BCI classification
        <https://hal.archives-ouvertes.fr/hal-00602700/>`_
        A. Barachant, S. Bonnet, M. Congedo and C. Jutten. 9th International
        Conference Latent Variable Analysis and Signal Separation
        (LVA/ICA 2010), LNCS vol. 6365, 2010, p. 629-636.
    """

    def __init__(self, metric="riemann", n_jobs=1):
        """Init."""
        self.metric = metric
        self.n_jobs = n_jobs

    def fit(self, X, y, sample_weight=None):
        """Fit (estimates) the centroids.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.
        y : ndarray, shape (n_matrices,)
            Labels for each matrix.
        sample_weight : None | ndarray, shape (n_matrices,), default=None
            Weights for each matrix. If None, it uses equal weights.

        Returns
        -------
        self : MDM instance
            The MDM instance.
        """
        self.metric_mean, self.metric_dist = check_metric(self.metric)
        self.classes_ = np.unique(y)

        if sample_weight is None:
            sample_weight = np.ones(X.shape[0])

        if self.n_jobs == 1:
            self.covmeans_ = [
                mean_covariance(X[y == ll], metric=self.metric_mean,
                                sample_weight=sample_weight[y == ll])
                for ll in self.classes_]
        else:
            self.covmeans_ = Parallel(n_jobs=self.n_jobs)(
                delayed(mean_covariance)(X[y == ll], metric=self.metric_mean,
                                         sample_weight=sample_weight[y == ll])
                for ll in self.classes_)

        self.covmeans_ = np.stack(self.covmeans_, axis=0)

        return self

    def _predict_distances(self, X):
        """Helper to predict the distance. Equivalent to transform."""
        n_centroids = len(self.covmeans_)

        if self.n_jobs == 1:
            dist = [distance(X, self.covmeans_[m], self.metric_dist)
                    for m in range(n_centroids)]
        else:
            dist = Parallel(n_jobs=self.n_jobs)(delayed(distance)(
                X, self.covmeans_[m], self.metric_dist)
                for m in range(n_centroids))

        dist = np.concatenate(dist, axis=1)
        return dist

    def predict(self, X):
        """Get the predictions.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.

        Returns
        -------
        pred : ndarray of int, shape (n_matrices,)
            Predictions for each matrix according to the closest centroid.
        """
        dist = self._predict_distances(X)
        return self.classes_[dist.argmin(axis=1)]

    def transform(self, X):
        """Get the distance to each centroid.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.

        Returns
        -------
        dist : ndarray, shape (n_matrices, n_classes)
            The distance to each centroid according to the metric.
        """
        return self._predict_distances(X)

    def fit_predict(self, X, y):
        """Fit and predict in one function."""
        self.fit(X, y)
        return self.predict(X)

    def predict_proba(self, X):
        """Predict proba using softmax of negative squared distances.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.

        Returns
        -------
        prob : ndarray, shape (n_matrices, n_classes)
            Probabilities for each class.
        """
        return softmax(-self._predict_distances(X) ** 2)

    def predict_proba_temperature(self, X, temperature=1):
        """Predict proba using softmax of distances divided by temperature.

            Parameters
            ----------
            temperature : value that determines the spread of probabilities in the softmax function
            X : ndarray, shape (n_matrices, n_channels, n_channels)
                Set of SPD matrices.

            Returns
            -------
            prob : ndarray, shape (n_matrices, n_classes)
                Probabilities for each class.
            """
        return softmax(self._predict_distances(X) / temperature)

