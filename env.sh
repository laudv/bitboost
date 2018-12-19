#!/bin/bash

export RUSTFLAGS="-C target-cpu=native -C opt-level=3 -C overflow-checks=off"
export RUST_LOG="spdyboost=debug"
