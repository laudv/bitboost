# Copyright (c) DTAI - KU Leuven - All rights reserved.
# Proprietary, do not copy or distribute without permission.
# Written by Laurens Devos, 2019

import os

from ctypes import *

import numpy as np
import pandas as pd

def _get_lib_dir():
    d = os.path.dirname(__file__)
    debug = os.path.join(d, "../../target/debug/libbitboost.so")
    release = os.path.join(d, "../../target/release/libbitboost.so")

    debug_if = os.path.isfile(debug)
    release_if = os.path.isfile(release)

    if debug_if and release_if:
        # use debug if newer, warn
        if os.path.getmtime(debug) > os.path.getmtime(release):
            print("WARNING: using newer BitBoost debug build")
            return debug
        else:
            return release
    elif debug_if:
        print("WARNING: using BitBoost debug build")
        return debug
    elif release_if:
        return release
    else:
        raise Exception("BitBoost library not found")

class BitBoost:
    _lib = CDLL(_get_lib_dir())
    _rust_numt_nbytes = _lib.bb_get_numt_nbytes
    _rust_numt_nbytes.argtypes = []
    _rust_numt_nbytes.restype = c_int
    _numt_nbytes = _rust_numt_nbytes()
    numt = c_double if _numt_nbytes == 8 else c_float
    numt_p = POINTER(numt)   

    _rust_alloc = _lib.bb_alloc
    _rust_alloc.argtypes = [c_int, c_int]
    _rust_alloc.restype = c_void_p

    _rust_dealloc = _lib.bb_dealloc
    _rust_dealloc.argtypes = [c_void_p]
    _rust_dealloc.restype = c_int

    _rust_set_fdata = _lib.bb_set_feature_data
    _rust_set_fdata.argtypes = [c_void_p, c_int, numt_p, c_int]
    _rust_set_fdata.restype = c_int

    _rust_set_config_field = _lib.bb_set_config_field
    _rust_set_config_field.argtypes = [c_void_p, c_char_p, c_char_p]
    _rust_set_config_field.restype = c_int

    _rust_train = _lib.bb_train
    _rust_train.argtypes = [c_void_p]
    _rust_train.restype = c_int

    _rust_predict = _lib.bb_predict
    _rust_predict.argtypes = [c_void_p, numt_p]
    _rust_predict.restype = c_int


    def __init__(self, nfeatures, nexamples):
        assert nfeatures > 0
        assert nexamples > 0
        self._ctx_ptr: c_void_p = c_void_p(0)
        self._nfeatures = nfeatures
        self._nexamples = nexamples

    def __enter__(self):
        assert not self._ctx_ptr
        self._ctx_ptr = self._rust_alloc(self._nfeatures, self._nexamples)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._check()
        self._rust_dealloc(self._ctx_ptr)
        self._ctx_ptr = c_void_p(0)

    def _check(self):
        if not self._ctx_ptr:
            raise Exception("use with-statement to ensure memory safety")

    def set_feature_data(self, feat_id, data, is_categorical):
        self._check()
        assert isinstance(data, np.ndarray)
        assert 0 <= feat_id and feat_id <= self._nfeatures
        assert isinstance(is_categorical, bool) 
        data_ptr = data.ctypes.data_as(self.numt_p)
        is_cat = 1 if is_categorical else 0
        self._rust_set_fdata(self._ctx_ptr, feat_id, data_ptr, is_cat)

    def set_data(self, data, cat_features = set()):
        self._check()
        assert isinstance(data, pd.DataFrame)
        assert data.shape[0] == self._nexamples
        assert data.shape[1] == self._nfeatures

        for feat_id in range(self._nfeatures):
            assert data.iloc[:, feat_id].dtype == self.numt

        for feat_id in range(self._nfeatures):
            col = data.iloc[:, feat_id].to_numpy(dtype=self.numt)
            is_cat = feat_id in cat_features
            self.set_feature_data(feat_id, col, is_cat)
    
    def set_target(self, data):
        self._check()
        self.set_feature_data(self._nfeatures, data, False)

    def set_config_field(self, name, value):
        self._check()
        n = c_char_p(bytes(str(name), "utf8"))
        v = c_char_p(bytes(str(value), "utf8"))
        self._rust_set_config_field(self._ctx_ptr, n, v)

    def set_config(self, values):
        self._check()
        assert isinstance(values, dict)
        for name, value in values.items():
            self.set_config_field(name, value)

    def train(self):
        self._check()
        self._rust_train(self._ctx_ptr)

    def predict(self):
        self._check()
        output = np.zeros(self._nexamples, dtype=self.numt)
        output_ptr = output.ctypes.data_as(self.numt_p)
        self._rust_predict(self._ctx_ptr, output_ptr)
        return output

    def write_model(self):
        raise Exception("not implemented")

    def read_model():
        raise Exception("not implemented")
