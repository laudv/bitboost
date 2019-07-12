"""
Lower-level RawBitBoost interface that interacts with the Rust code though a C ABI.
"""

import sys
import os
import timeit

# use local python package rather than the system install
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../python"))

from bitboost.bitboost import RawBitBoost
import numpy as np
import sklearn.metrics

nfeatures = 200
nexamples = 1000000
np.random.seed(0)
data = np.random.choice(np.array([0.0, 1.0, 2.0], dtype=RawBitBoost.numt),
                        size=(nexamples * 2, nfeatures))
target = ((data[:, 0] > 1.0)
        & (data[:, 1] > 1.0)
        & (data[:, 2] != 2.0)
        | (data[:, 3] == 1.0)).astype(RawBitBoost.numt)
dtrain, ttrain = data[0:nexamples, :], target[0:nexamples]
dtest, ttest = data[nexamples:, :], target[nexamples:]

bb = RawBitBoost(nfeatures, nexamples)
bb.set_config({
    "objective": "hinge",
    "discr_nbits": 2,
    "max_tree_depth": 5,
    "learning_rate": 0.2,
    "niterations": 10,
    "metric_frequency": 0,
    "metrics": "binary_error"})
bb.set_data(dtrain, cat_features = set(range(nfeatures)))
bb.set_target(ttrain)
bbt = timeit.timeit(lambda: bb.train(), number=5)
print(f"bitboost: {bbt/10} sec")
predictions = bb.predict()

balance = target.sum() / target.shape[0]
print("class balance: {:.2} vs. {:.2}".format(balance, 1-balance))
acc = sklearn.metrics.accuracy_score(ttrain==1.0, predictions > 0)
print(f"bit train accuracy: {acc}")

bb.set_data(dtest)
predictions = bb.predict()
acc = sklearn.metrics.accuracy_score(ttest==1.0, predictions > 0)
print(f"bit test accuracy: {acc}")



# xgb
import xgboost as xgb

dtrain = xgb.DMatrix(dtrain, label=ttrain)
param = {
        'tree_method': 'hist',
        'objective': 'binary:logistic',
        'max_depth': 5,
        'num_round': 10,
        'learning_rate': 0.2,
        #'nthread': 1
}
bst = xgb.train(param, dtrain)

xgbt = timeit.timeit(lambda: xgb.train(param, dtrain), number=5)
print(f"xgboost: {xgbt/10} sec")

predictions = bst.predict(dtrain, output_margin=True)
acc = sklearn.metrics.accuracy_score(ttrain==1.0, predictions > 0)
print(f"xgb train accuracy: {acc}")

predictions = bst.predict(xgb.DMatrix(dtest), output_margin=True)
acc = sklearn.metrics.accuracy_score(ttest==1.0, predictions > 0)
print(f"xgb test accuracy: {acc}")
