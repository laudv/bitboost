import sys
import os
import timeit

sys.path.append(os.path.join(os.path.dirname(__file__), "../python"))

from bitboost import BitBoostRegressor
import numpy as np
import pandas as pd
import sklearn.metrics

nfeatures = 5
nexamples = 10000
data = np.random.choice(np.array([0.0, 1.0, 2.0], dtype=BitBoostRegressor.numt),
                        size=(nexamples * 2, nfeatures))
target = (1.22 * (data[:, 0] > 1.0)
        + 0.65 * (data[:, 1] > 1.0)
        + 0.94 * (data[:, 2] != 2.0)
        + 0.13 * (data[:, 3] == 1.0)).astype(BitBoostRegressor.numt)
dtrain, ytrain = data[0:nexamples, :], target[0:nexamples]
dtest, ytest = data[nexamples:, :], target[nexamples:]

bit = BitBoostRegressor()
bit.objective = "l2"
bit.discr_nbits = 4
bit.max_tree_depth = 5
bit.learning_rate = 0.5
bit.niterations = 50
bit.categorical_features = list(range(nfeatures))

bit.fit(pd.DataFrame(dtrain), ytrain)
train_pred = bit.predict(pd.DataFrame(dtrain))
test_pred = bit.predict(pd.DataFrame(dtest))

train_acc = sklearn.metrics.mean_absolute_error(ytrain, train_pred)
test_acc = sklearn.metrics.mean_absolute_error(ytest, test_pred)
print(f"bit train accuracy: {train_acc}")
print(f"bit test accuracy: {test_acc}")

