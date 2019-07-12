[![Build Status](https://travis-ci.org/laudv/bitboost.svg?branch=master)](https://travis-ci.org/laudv/bitboost)

# BitBoost

BitBoost is a gradient boosting decision tree model similar to [XGBoost],
[LightGBM], and [CatBoost]. Unlike these systems, BitBoost uses bitslices to
represent discretized gradients and bitsets to represent the data vectors and
the instance lists, with the goal of improving learning speed.

***Note:*** this is an experimental system, and

 - BitBoost does not (yet) support multi-class classification,
 - BitBoost does not (yet) support proper multi-threading,
 - BitBoost does not (yet) effectively handle sparse features,
 - BitBoost works best for low-cardinality categorical features,
 - BitBoost can handle high-cardinality categorical and numerical features efficiently given that (1) there are not too many and (2) only coarse-grained splits are required on those features, i.e., we can have high `sample_freq` and low `max_nbins` paramater values.
 
 Specifically, BitBoost will most likely perform worse on fully numerical datasets. In that case, use LightGBM, XGBoost or CatBoost instead.

## Compiling

BitBoost is implemented in Rust and uses the standard Rust tools, `cargo` and
`rustc`.

 - Make sure you have Rust 2018 edition installed.
 - Clone this repository.
 - Tell `rustc` to generate efficient [AVX2 instructions][AVX2] (ensure you have a AVX2
   capable CPU):
   ```
   export RUSTFLAGS="-C target-cpu=native"
   ```
 - Compile the code:
   ```
   cargo build --release
   ```



## Using BitBoost from Python

```python
import numpy as np
import sklearn.metrics

from bitboost import BitBoostRegressor

# Generate some categorical data
nfeatures = 5
nexamples = 10000
data = np.random.choice(np.array([0.0, 1.0, 2.0], dtype=BitBoostRegressor.numt),
                        size=(nexamples * 2, nfeatures))
target = (1.22 * (data[:, 0] > 1.0)
        + 0.65 * (data[:, 1] > 1.0)
        + 0.94 * (data[:, 2] != 2.0)
        + 0.13 * (data[:, 3] == 1.0)).astype(BitBoostRegressor.numt)

# Run BitBoost
bit = BitBoostRegressor(
    objective="l2", discr_nbits=4, max_tree_depth=5, learning_rate=0.5,
    niterations=20, categorical_features=list(range(nfeatures)))
bit.fit(data, target)

train_acc = sklearn.metrics.mean_absolute_error(target, bit.predict(data))
```

## Running from the Command Line

Use the `bitboost` binary to run BitBoost from the command line:


```
./target/release/run_bitboost boost \
    train=/path/to/train.csv \
    test=/path/to/test.csv \
    objective=binary \
    niterations=10 \
    learning_rate=0.5 \
    metrics=binary_error,binary_loss \
    categorical_features=0,1,2,3,4 \
    sample_freq=10 \
    discr_nbits=8 \
    max_nbins=16
```


## Python Interface

BitBoost has a [Scikit-learn](https://scikit-learn.org/stable/) interface. A number
of examples are provided in the [examples](examples) folder.


## Parameters

All the parameters can be found in [src/config.rs](src/config.rs). The supported
objectives are in [src/objective.rs](src/objective.rs).



# Paper: Fast Gradient Boosting Decision Trees with Bit-Level Data Structures

Check out the [experiments](https://github.com/laudv/bitboost/tree/experiments)
branch to see the experimental setup, or quickly navigate to the results for:

 - [Allsate](https://github.com/laudv/bitboost/blob/experiments/experiments/allstate/run-allstate.ipynb)
 - [CoverType](https://github.com/laudv/bitboost/blob/experiments/experiments/covtype/run-covtype.ipynb)
 - [Binary-MNIST](https://github.com/laudv/bitboost/blob/experiments/experiments/bin-mist/run-bin-mnist.ipynb)
 - [YouTube](https://github.com/laudv/bitboost/blob/experiments/experiments/youtube/run-youtube.ipynb)




[XGBoost]: https://xgboost.readthedocs.io
[LightGBM]: https://lightgbm.readthedocs.io
[CatBoost]: https://catboost.ai
[AVX2]: https://en.wikipedia.org/wiki/Advanced_Vector_Extensions#Advanced_Vector_Extensions_2
