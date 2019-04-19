

from sklearn.base import BaseEstimator
from .bitboost import RawBitBoost

# https://scikit-learn.org/stable/modules/generated/sklearn.utils.estimator_checks.check_estimator.html#sklearn.utils.estimator_checks.check_estimator
class BitBoost(BaseEstimator):
    """
    BitBoost base estimator.
    """
    __doc__ += RawBitBoost.__doc__

    numt = RawBitBoost.numt
    numt_p = RawBitBoost.numt_p

    def __init__():
        pass

    def fit(data, target):
        pass

    def predict(data):
        pass
