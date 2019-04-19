import sys
import os
import timeit

sys.path.append(os.path.join(os.path.dirname(__file__), "../python"))

from bitboost import BitBoostClassifier
import numpy as np
import pandas as pd
import sklearn.metrics

nfeatures = 5
nexamples = 100
data = np.random.choice(np.array([0.0, 1.0, 2.0], dtype=BitBoostClassifier.numt),
                        size=(nexamples * 2, nfeatures))
target = ((data[:, 0] > 1.0)
        & (data[:, 1] > 1.0)
        & (data[:, 2] != 2.0)
        | (data[:, 3] == 1.0)).astype(BitBoostRegressor.numt)
dtrain, ytrain = data[0:nexamples, :], target[0:nexamples]
dtest, ytest = data[nexamples:, :], target[nexamples:]

bit = BitBoostClassifier()
bit.objective = "hinge"
bit.discr_nbits = 2
bit.max_tree_depth = 5
bit.learning_rate = 0.2
bit.niterations = 10
bit.categorical_features = list(range(nfeatures))

bit.fit(pd.DataFrame(dtrain), ytrain)
train_pred = bit.predict(pd.DataFrame(dtrain))
test_pred = bit.predict(pd.DataFrame(dtest))

train_acc = sklearn.metrics.accuracy_score(ytrain==1.0, train_pred > 0)
test_acc = sklearn.metrics.accuracy_score(ytest==1.0, test_pred > 0)
print(f"bit train accuracy: {train_acc}")
print(f"bit test accuracy: {test_acc}")

