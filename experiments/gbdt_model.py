import time, sys, os
import numpy as np
import sklearn as skl

import xgboost as xgb
import lightgbm as lgb
import catboost as cat


BITBOOST_PYTHON_PATH = os.path.join(os.path.dirname(__file__), "../python")
if BITBOOST_PYTHON_PATH not in sys.path:
    print("appending bitboost to path", BITBOOST_PYTHON_PATH)
    sys.path.append(BITBOOST_PYTHON_PATH)
from bitboost.bitboost import RawBitBoost

class Dataset:
    def __init__(self, df, target_name):
        self.target = df[target_name]
        df = df.drop(columns=target_name)
        self.data = df

class GBDTModel:
    def __init__(self, name):
        self._name = name
        self._params = {}
        self._train = None
        self._test = None

    def name(self):
        return self._name

    def set_params(self, params):
        self._params = params

    def set_data(self, train, test):
        self._train = train
        self._test = test

    def train(self):
        """ Returns (time to train, eval train, eval test) """
        raise Exception("not implemented")

    def compute_metric(self, actual, pred):
        metric = self._params["metric"]
        if metric == "logloss":
            return skl.metrics.log_loss(actual, pred)
        if metric == "error":
            return 1.0 - skl.metrics.accuracy_score(np.array(actual)==1.0,
                                                    np.array(pred) > 0.0)
        if metric == "error01":
            return 1.0 - skl.metrics.accuracy_score(np.array(actual)==1.0,
                                                    np.array(pred) > 0.5)
        if metric == "rmse":
            return np.sqrt(skl.metrics.mean_squared_error(actual, pred))
        if metric == "mae":
            return skl.metrics.mean_absolute_error(actual, pred)
        if metric == "mae_exp":
            return skl.metrics.mean_absolute_error(np.exp(actual), np.exp(pred))
        if metric == "med_ae":
            return skl.metrics.median_absolute_error(actual, pred)
        if metric == "med_ae_exp":
            return skl.metrics.median_absolute_error(np.exp(actual), np.exp(pred))

class XGBModel(GBDTModel):
    def __init__(self):
        super().__init__("xgb")
        self._xgbparams = {} 
        self._xgbtrain = None
        self._xgbtest = None

    def set_params(self, params):
        super().set_params(params)

        objective_map = { "binary": "reg:logistic", "reg_l2": "reg:linear",
                "hinge": "binary:hinge" }

        if not params["objective"] in objective_map:
            raise Exception("objective {} not supported".format(params["objective"]))

        self._xgbparams = {
            "booster":              "gbtree",
            "verbosity":            0,
            "nthread":              params["nthreads"],
            "learning_rate":        params["learning_rate"],
            "gamma":                params["min_gain"], # min loss required to make a further split
            "max_depth":            params["max_depth"],
            "min_child_weight":     1.0, # min sum instance weight (hessian) in leaf
            "max_delta_step":       0.0, # maximum leaf output value? unbalanced logistic regression
            "subsample":            params["example_fraction"],
            "colsample_bytree":     params["feature_fraction"],
            "reg_lambda":           params["l2_reg"],
            "reg_alpha":            params["l1_reg"],
            "tree_method":          params["xgb_tree_method"],
            "scale_pos_weight":     1.0, # balance pos/neg weights for unbalanced classes; default
            "refresh_leaf":         True, #default
            "grow_policy":          "depthwise", # default
            "max_bin":              255, # match lightgbm
            "num_parallel_tree":    1,

            "objective":        objective_map[params["objective"]],
            "seed":             params["random_seed"],
        }

    def set_data(self, train, test):
        super().set_data(train, test)
        missing = self._params["xgb_missing"]
        self._xgbtrain = xgb.DMatrix(train.data, label=train.target, missing=missing)
        self._xgbtest  = xgb.DMatrix(test.data, label=test.target, missing=missing)

    def train(self):
        start_time = time.process_time()
        bst = xgb.train(self._xgbparams, self._xgbtrain, self._params["niterations"])
        train_time = time.process_time() - start_time

        pred_train = bst.predict(self._xgbtrain, output_margin=True)
        pred_test  = bst.predict(self._xgbtest,  output_margin=True)

        metric_train = self.compute_metric(self._train.target, pred_train)
        metric_test  = self.compute_metric(self._test.target,  pred_test)

        return (train_time, metric_train, metric_test)

class LGBModel(GBDTModel):
    def __init__(self):
        super().__init__("xgb")
        self._lgbparams = {} 
        self._lgbtrain = None
        self._lgbtest = None

    def set_params(self, params):
        super().set_params(params)

        objective_map = { "binary": "binary", "reg_l1": "regression_l1",
                "reg_l2": "regression_l2", "huber": "huber" }

        if not params["objective"] in objective_map:
            raise Exception("objective {} not supported".format(params["objective"]))

        self._lgbparams = {
            "objective":            objective_map[params["objective"]],
            "boosting":             params["lgb_boosting_type"],
            "learning_rate":        params["learning_rate"],
            "num_leaves":           1 << params["max_depth"] - 1,
            "max_depth":            params["max_depth"],
            "tree_learner":         "serial",
            "num_threads":          params["nthreads"],
            "device_type":          "cpu",
            "seed":                 params["random_seed"],
            "min_data_in_leaf":     1,
            "min_sum_hessian":      1.0,
            "bagging_fraction":     params["example_fraction"],
            "bagging_freq":         params["lgb_sample_freq"],
            "bagging_seed":         params["random_seed"] + 79,
            "feature_fraction":     params["feature_fraction"],
            "feature_fraction_seed": params["random_seed"] + 88,
            "max_delta_step":       0.0, # alias max_leaf_output, match xgboost
            "lambda_l1":            params["l1_reg"],
            "lambda_l2":            params["l2_reg"],
            "min_split_gain":       params["min_gain"],
            "alpha":                params["huber_alpha"],

            "min_data_per_group":   100, # default, min number of examples with categorical value
            "cat_l2":               10.0, # default
            "cat_smooth":           10.0, # default
            "max_cat_to_onehot":    4, # default, match catboost

            "verbosity":            -1,
            "max_bin":              255, # match xgboost
            "enable_bundle":        params["lgb_efb"], # Exclusive Feature Bundling (EFB)
            "enable_sparse":        params["lgb_sparse"],
            "sparse_threshold":     0.8,   # default,
            #"categorical_feature":  params["categorical"], # emits warning

            "use_missing":          False,
            "zero_as_missing":      False,
        }

    def set_data(self, train, test):
        super().set_data(train, test)
        self._lgbtrain = lgb.Dataset(train.data, label=train.target)

    def train(self):
        start_time = time.process_time()
        bst = lgb.train(self._lgbparams, self._lgbtrain, self._params["niterations"])
        train_time = time.process_time() - start_time

        pred_train = bst.predict(self._train.data, raw_score=True)
        pred_test  = bst.predict(self._test.data,  raw_score=True)

        metric_train = self.compute_metric(self._train.target, pred_train)
        metric_test  = self.compute_metric(self._test.target,  pred_test)

        return (train_time, metric_train, metric_test)

class CatModel(GBDTModel):
    def __init__(self):
        super().__init__("cat")
        self._catparams = {} 
        self._cattrain = None
        self._cattest = None

    def set_params(self, params):
        super().set_params(params)

        objective_map = { "binary": "Logloss", "reg_l1": "MAE", "reg_l2": "RMSE" }

        if not params["objective"] in objective_map:
            raise Exception("objective {} not supported".format(params["objective"]))

        self._catparams = {
            "objective":            objective_map[params["objective"]],
            "iterations":           params["niterations"],
            "learning_rate":        params["learning_rate"],
            "random_seed":          params["random_seed"],
            "reg_lambda":           params["l2_reg"],
            "bootstrap_type":       "Bernoulli",
            "subsample":            params["example_fraction"],
            "sampling_frequency":   "PerTree",
            "colsample_bylevel":    params["feature_fraction"], # no bytree option
            "max_depth":            params["max_depth"],
            "one_hot_max_size":     4, # default, match lightgbm
            "max_bin":              255, # match lightgbm, xgboost (alias border_count)
            "boosting_type":        params["cat_boosting_type"],
            "thread_count":         params["nthreads"],
            "task_type":            "CPU",
            "logging_level":        "Silent",
            "train_dir":            "/tmp/catboost-tmp",
        }

    def set_data(self, train, test):
        super().set_data(train, test)
        self._cattrain = cat.Pool(train.data, label=train.target,
                cat_features=self._params["categorical"])

    def train(self):
        start_time = time.process_time()
        bst = cat.CatBoost(self._catparams)
        bst.fit(self._cattrain)
        train_time = time.process_time() - start_time

        pred_train = bst.predict(self._train.data, prediction_type="RawFormulaVal")
        pred_test  = bst.predict(self._test.data,  prediction_type="RawFormulaVal")

        metric_train = self.compute_metric(self._train.target, pred_train)
        metric_test  = self.compute_metric(self._test.target,  pred_test)

        return (train_time, metric_train, metric_test)

class BitModel(GBDTModel):
    def __init__(self):
        super().__init__("bit")
        self._bitparams = {} 
        self._bittrain = None
        self._bittest = None
        self._bittrain_target = None

    def set_params(self, params):
        super().set_params(params)

        objective_map = { "binary": "binary", "reg_l1": "l1", "reg_l2": "l2",
                "huber": "huber", "hinge": "hinge" }

        self._bitparams = {
            "objective":            objective_map[params["objective"]],
            "metric_frequency":     0,
            "categorical_features": ",".join(map(str, params["categorical"])),
            "niterations":          params["niterations"],
            "learning_rate":        params["learning_rate"],
            "reg_lambda":           params["l2_reg"],
            "min_examples_leaf":    1, # match lgbm
            "min_gain":             params["min_gain"],
            "max_nbins":            params["bit_max_nbins"],
            "discr_nbits":          params["bit_discr_nbits"],
            "binary_gradient_bound": params["bit_binary_grad_bound"],
            "max_tree_depth":       params["max_depth"],
            "compression_threshold": params["bit_compr_threshold"],
            "random_seed":          params["random_seed"],
            "feature_fraction":     params["feature_fraction"],
            "example_fraction":     params["example_fraction"],
            "sample_freq":          params["bit_sample_freq"],
            "huber_alpha":          params["huber_alpha"],
        }

    def set_data(self, train, test):
        super().set_data(train, test)
        self._bittrain = train.data.to_numpy(dtype=np.float32)
        self._bittest = test.data.to_numpy(dtype=np.float32)
        self._bittrain_target = train.target.to_numpy(dtype=np.float32)

    def train(self):
        nexamples, nfeatures = self._bittrain.shape
        cat_features = set(self._params["categorical"])

        bb = RawBitBoost(nfeatures, nexamples)
        bb.set_config(self._bitparams)
        bb.set_data(self._bittrain, cat_features=cat_features)
        bb.set_target(self._bittrain_target)

        start_time = time.process_time()
        bb.train()
        train_time = time.process_time() - start_time

        pred_train = bb.predict()
        bb.set_data(self._bittest, cat_features=cat_features)
        pred_test  = bb.predict()

        metric_train = self.compute_metric(self._train.target, pred_train)
        metric_test  = self.compute_metric(self._test.target,  pred_test)

        return (train_time, metric_train, metric_test)
