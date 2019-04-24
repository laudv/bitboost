import numpy as np

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.estimator_checks import check_estimator

from .bitboost import RawBitBoost, gen_init_fun

# https://github.com/scikit-learn-contrib/project-template/
class BitBoost(BaseEstimator):
    """ BitBoost base estimator.
    """

    numt = RawBitBoost.numt
    numt_p = RawBitBoost.numt_p

    __init__ = gen_init_fun(RawBitBoost.config_params, __file__)

    def fit(self, X, y):
        """ Fit a BitBoost model to training examples (X, y).

        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            The training input samples.
        y : {pandas.Series}, shape (n_samples,)
            The target values (class labels in classification, real numbers in
            regression).

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y(X, y, accept_sparse=False, dtype=self.numt,
                         order="F", # column-major
                         warn_on_dtype=True)

        nexamples, nfeatures = X.shape

        self._bitboost = RawBitBoost(nfeatures, nexamples)
        self._bitboost.set_config(self.get_params())
        self._bitboost.set_data(X, self.categorical_features)
        self._bitboost.set_target(y)

        self._bitboost.train()

        self._is_fitted = True
        return self

    def predict(self, X):
        """ Predict values for given input data.

        Parameters
        ----------
        X : {pandas.DataFrame}, shape (n_samples, n_features)
            The training input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            Returns an array of predictions.
        """
        X = check_array(X, accept_sparse=False)
        check_is_fitted(self, "_is_fitted")

        self._bitboost.set_data(X)
        return self._bitboost.predict()

    def _check_sklearn_estimator():
        check_estimator(BitBoost)


class BitBoostRegressor(BitBoost):
    pass

class BitBoostClassifier(BitBoost):
    pass
