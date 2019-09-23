[![Build Status](https://travis-ci.org/laudv/bitboost.svg?branch=master)](https://travis-ci.org/laudv/bitboost)

# BitBoost

BitBoost is a gradient boosting decision tree model similar to [XGBoost],
[LightGBM], and [CatBoost]. Unlike these systems, BitBoost uses bitslices to
represent discretized gradients and bitsets to represent the data vectors and
the instance lists, with the goal of improving learning speed.

BitBoost outperforms the other boosting systems in terms of training time when a significant number of input features are categorical and have only few possible values (i.e., low cardinality). Here are some numbers:

Time (seconds):

|   | [Allstate][dsa] | [Covtype1][dsc] | [Covtype2][dsc] | [Bin-MNIST][dsm] | [YouTube][dsy] |
|---|----------|----------|----------|-----------|---------|
| BitBoost Accurate | 4.8   | 17.1  | 10.7  | 4.5   | 14.3 |
| BitBoost Fast     | 1.0   | 5.4   | 7.2   | 1.9   | 2.5  |
| LightGBM          | 12.3  | 24.1  | 21.0  | 24.8  | 35.0 |
| XGBoost           | 11.5  | 37.0  | 35.3  | 24.7  | 24.9 |
| CatBoost          | 82.6  | 58.1  | 52.9  | 16.5  | 33.6 |


Accuracy (MAE, Error%, Error%, Error%, MAE):

|   | [Allstate][dsa] | [Covtype1][dsc] | [Covtype2][dsc] | [Bin-MNIST][dsm] | [YouTube][dsy] |
|---|----------|----------|----------|-----------|---------|
| BitBoost Accurate | 1159  | 12.0  | 0.79  | 2.78  | 0.07 |
| BitBoost Fast     | 1194  | 14.9  | 1.02  | 3.52  | 0.12 |
| LightGBM          | 1156  | 11.9  | 0.71  | 2.86  | 0.07 |
| XGBoost           | 1157  | 10.8  | 0.63  | 2.66  | 0.07 |
| CatBoost          | 1167  | 13.1  | 0.91  | 3.23  | 0.11 |

Click the column labels, or read the [paper for][paper] more information.


***Note:*** this is an experimental system, and

 - BitBoost does not (yet) support multi-class classification,
 - BitBoost does not (yet) support proper multi-threading,
 - BitBoost does not (yet) effectively handle sparse features,
 - BitBoost works best for low-cardinality categorical features,
 - BitBoost can handle high-cardinality categorical and numerical features efficiently given that (1) there are not too many and (2) only coarse-grained splits are required on those features, i.e., we can have high `sample_freq` and low `max_nbins` paramater values.
 
Specifically, BitBoost will most likely perform worse on fully numerical datasets. In that case, use LightGBM, XGBoost or CatBoost instead.

## License

&copy; [DTAI Research Group][dtai] - [KU Leuven][kul].
Licensed under the Apache License 2.0.

## Citing

Please cite [this paper][paper]: 

Devos, L., Meert, W., & Davis, J. (2019). Fast Gradient Boosting Decision Trees with Bit-Level Data Structures. In *Proceedings of ECML PKDD*. Springer.

## Compiling

BitBoost is implemented in stable Rust and uses the [standard Rust tools][rustup], `cargo` and
`rustc`.

 - Make sure you have Rust 2018 edition installed, that is, Rust 1.31 or higher.
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

BitBoost is not available on pip just yet. However, you can install a pip package on your local Linux system as follows.

First, ensure you have [Rust][rustup] installed. Activate the Python3 environment of your liking, and run:
```
cd <bitboost-repo>/python
python setup.py install [--user]
```
Use `--user` if you don't have write access to your site-packages directory. Test your installation with the following code snippet:

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

BitBoost has a [Scikit-learn](https://scikit-learn.org/stable/) interface. A number
of examples are provided in the [examples](examples) folder.

## Running from the Command Line

Use the `run_bitboost` binary to run BitBoost from the command line:


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

This only supports CSV input files.



## Parameters

All the parameters can be found in [src/config.rs](src/config.rs). The supported
objectives are in [src/objective.rs](src/objective.rs).

In Python, you can refer to the parameter documentation as follows:

```python
import bitboost

help(bitboost.BitBoost)
```



# Paper: Fast Gradient Boosting Decision Trees with Bit-Level Data Structures

Check out the [experiments](https://github.com/laudv/bitboost/tree/experiments)
branch to see the experimental setup, or quickly navigate to the results for:

 - [Allsate][dsa]
 - [CoverType][dsc]
 - [Binary-MNIST][dsm]
 - [YouTube][dsy]



[dsa]: https://github.com/laudv/bitboost/blob/experiments/experiments/allstate/run-allstate.ipynb
[dsc]: https://github.com/laudv/bitboost/blob/experiments/experiments/covtype/run-covtype.ipynb
[dsm]: https://github.com/laudv/bitboost/blob/experiments/experiments/bin-mist/run-bin-mnist.ipynb
[dsy]: https://github.com/laudv/bitboost/blob/experiments/experiments/youtube/run-youtube.ipynb

[rustup]: https://rustup.rs
[XGBoost]: https://xgboost.readthedocs.io
[LightGBM]: https://lightgbm.readthedocs.io
[CatBoost]: https://catboost.ai
[AVX2]: https://en.wikipedia.org/wiki/Advanced_Vector_Extensions#Advanced_Vector_Extensions_2
[paper]: https://scholar.google.be/scholar?q=Fast+Gradient+Boosting+Decision+Trees+with+Bit-Level+Data+Structures+Devos+Meert+Davis
[dtai]: https://dtai.cs.kuleuven.be
[kul]: https://www.kuleuven.be/english
