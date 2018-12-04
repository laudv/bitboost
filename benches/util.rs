use rand::prelude::*;
use rand::distributions::{Uniform, Bernoulli};

use spdyboost::bits::{BitSlice2, BitSlice4};
use spdyboost::bits::ValueStore;
use spdyboost::bits::FullBitSet;

pub fn gen_select_full(size: usize, frac1: f64) -> FullBitSet {
    let mut rng = thread_rng();
    let dist = Bernoulli::new(frac1);
    FullBitSet::from_bool_iter(rng.sample_iter(&dist).take(size))
}

pub fn gen_bitslice2(size: usize) -> BitSlice2 {
    let mut rng = thread_rng();
    let dist = Uniform::new(0u64, 0xFFFFFFFFu64);
    let mut slice = BitSlice2::new(size);
    for i in 0..size {
        slice.set_value(i, dist.sample(&mut rng)&0b1111);
    }
    slice
}

pub fn gen_bitslice4(size: usize) -> BitSlice4 {
    let mut rng = thread_rng();
    let dist = Uniform::new(0u64, 0xFFFFFFFFu64);
    let mut slice = BitSlice4::new(size);
    for i in 0..size {
        slice.set_value(i, dist.sample(&mut rng)&0b1111);
    }
    slice
}
