
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../python"))

from bitboost import BitBoost
import numpy as np
import pandas as pd

nfeatures = 5
nexamples = 10
with BitBoost(nfeatures, nexamples) as bb:
    data = np.random.choice(np.array([1.0, 2.0, 3.0], dtype=BitBoost.numt),
                            size=(nexamples, nfeatures))
    target = data[:, 1] + data[:, 2]
    bb.set_data(pd.DataFrame(data=data))
    bb.set_target(target)
    bb.set_config({
        "objective": "l2",
        "discr_nbits": 4,
        "max_depth": 5,
        "learning_rate": 0.5,
        "niterations": 15})
    bb.train()
    predictions = bb.predict()

    error = sum((target-predictions) ** 2) / nexamples
    print(f"train error: {error}")
