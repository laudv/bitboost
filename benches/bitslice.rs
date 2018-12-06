#[macro_use]
extern crate criterion;
extern crate rand;
extern crate spdyboost;

use criterion::Criterion;
use spdyboost::bits::{BitSet, BitSlice};

const NVALUES: usize = 1_000_000;

fn bench_bitslice_sum1(c: &mut Criterion) {
    let slice = BitSlice::random(NVALUES, 1);
    c.bench_function("bitslice_sum1", move |b| {
        b.iter(|| slice.sum())
    });
}

fn bench_bitslice_sum2(c: &mut Criterion) {
    let slice = BitSlice::random(NVALUES, 2);
    c.bench_function("bitslice_sum2", move |b| {
        b.iter(|| slice.sum())
    });
}

fn bench_bitslice_sum4(c: &mut Criterion) {
    let slice = BitSlice::random(NVALUES, 4);
    c.bench_function("bitslice_sum4", move |b| {
        b.iter(|| slice.sum())
    });
}

fn bench_bitslice_sum1_masked(c: &mut Criterion) {
    let slice = BitSlice::random(NVALUES, 1);
    let mask = BitSet::random(NVALUES, 0.25);
    c.bench_function("bitslice_sum_masked1", move |b| {
        b.iter(|| slice.sum_masked(&mask))
    });
}

fn bench_bitslice_sum2_masked(c: &mut Criterion) {
    let slice = BitSlice::random(NVALUES, 2);
    let mask = BitSet::random(NVALUES, 0.25);
    c.bench_function("bitslice_sum_masked2", move |b| {
        b.iter(|| slice.sum_masked(&mask))
    });
}

fn bench_bitslice_sum4_masked(c: &mut Criterion) {
    let slice = BitSlice::random(NVALUES, 4);
    let mask = BitSet::random(NVALUES, 0.25);
    c.bench_function("bitslice_sum_masked4", move |b| {
        b.iter(|| slice.sum_masked(&mask))
    });
}

criterion_group!(benches,
                 bench_bitslice_sum1_masked,
                 bench_bitslice_sum2_masked,
                 bench_bitslice_sum4_masked,
                 );
criterion_main!(benches);
