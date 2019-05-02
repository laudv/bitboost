/*
 * Copyright (c) DTAI - KU Leuven - All rights reserved.
 * Proprietary, do not copy or distribute without permission.
 * Written by Laurens Devos, 2019
*/

#[macro_export]
macro_rules! safety_check {
    //($assertion:expr) => { assert!($assertion); } // enabled
    ($assertion:expr) => {} // disabled
}

pub type NumT = f32; // numeric type
pub type CatT = u32; // unsigned int of same size as NumT
pub const EPSILON: NumT = std::f32::EPSILON;
pub const POS_INF: NumT = std::f32::INFINITY;
pub const NEG_INF: NumT = std::f32::NEG_INFINITY;
pub fn into_cat(x: NumT) -> CatT { debug_assert!(x >= 0.0 && x.round() == x); x as CatT }

pub mod config;
pub mod data;
pub mod dataset;
pub mod bitblock;
pub mod simd;
pub mod harley_seal;
pub mod bitset;
pub mod bitslice;
pub mod slice_store;
pub mod tree;
pub mod objective;
pub mod binner;
pub mod metric;
pub mod boost;
pub mod c_api;
