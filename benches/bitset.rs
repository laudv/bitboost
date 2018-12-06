#[macro_use]
extern crate criterion;
extern crate rand;
extern crate spdyboost;

use criterion::{Criterion, Bencher, Fun};

use spdyboost::bits::BitSet;

mod util;

const NVALUES: usize = 1_000_000;

fn bench_bitset_popcnt(b: &mut Bencher, frac1: &f64) {
    let set = BitSet::random(NVALUES, *frac1);
    b.iter(|| set.count_ones_popcnt())
}

fn bench_bitset_avx2(b: &mut Bencher, frac1: &f64) {
    let set = BitSet::random(NVALUES, *frac1);
    b.iter(|| set.count_ones_avx2())
}

fn bench_bitset_count_ones(c: &mut Criterion) {
    let popcnt = Fun::new("count_ones_popcnt", bench_bitset_popcnt);
    let avx2 = Fun::new("count_ones_avx2", bench_bitset_avx2);
    let funs = vec![popcnt, avx2];

    c.bench_functions("bitset_count_ones", funs, 0.25f64);
}

criterion_group!(benches,
                 bench_bitset_count_ones,
                 );
criterion_main!(benches);
