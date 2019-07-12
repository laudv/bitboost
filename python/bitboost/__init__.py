# Copyright (c) DTAI - KU Leuven - All rights reserved.
# Proprietary, do not copy or distribute without permission.
# Written by Laurens Devos, 2019

name = "bitboost"

from .sklearn import BitBoost, BitBoostRegressor, BitBoostClassifier

__author__ = "Laurens Devos"
__copyright__ = "Copyright (c) DTAI - KU Leuven"
#__license__ = "???" # TODO license
__all__ = [
    "BitBoost",
    "BitBoostRegressor",
    "BitBoostClassifier"
]
