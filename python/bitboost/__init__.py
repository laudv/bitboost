# Copyright (c) DTAI - KU Leuven - All rights reserved.
# Proprietary, do not copy or distribute without permission.
# Written by Laurens Devos, 2019

name = "bitboost"

from .bitboost import RawBitBoost
from .sklearn import BitBoost

__all__ = [
    "BitBoost"
]
