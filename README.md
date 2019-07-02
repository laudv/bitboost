[![Build Status](https://travis-ci.org/laudv/bitboost.svg?branch=master)](https://travis-ci.org/laudv/bitboost)

# BitBoost

BitBoost is a gradient boosting decision tree model similar to [XGBoost],
[LightGBM], and [CatBoost]. Unlike these systems, BitBoost uses bitslices to
represent discretized gradients and bitsets to represent the data vectors and
the instance lists, with the goal of improving learning speed.

***Note:*** this is an experimental system.

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


## Running from the Command Line

Use the `bitboost` binary to run BitBoost from the command line:


```
./target/release/bitboost boost \
    train=/path/to/train.csv \
    test=/path/to/test.csv \
    objective=binary \
    niterations=10 \
    learning_rate=0.5 \
    metrics=binary_error,binary_loss \
    categorical_features=0,1,2,3,4 \
    sample_freq=10
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
