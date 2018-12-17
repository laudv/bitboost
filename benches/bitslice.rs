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

fn bench_bitslice_sum1_masked2(c: &mut Criterion) {
    let slice = BitSlice::random(NVALUES, 1);
    let mask1 = BitSet::random(NVALUES, 0.25);
    let mask2 = BitSet::random(NVALUES, 0.50);
    c.bench_function("bitslice_sum_masked2_1", move |b| {
        b.iter(|| slice.sum_masked2(&mask1, &mask2))
    });
}

fn bench_bitslice_sum2_masked2(c: &mut Criterion) {
    let slice = BitSlice::random(NVALUES, 2);
    let mask1 = BitSet::random(NVALUES, 0.25);
    let mask2 = BitSet::random(NVALUES, 0.50);
    c.bench_function("bitslice_sum_masked2_2", move |b| {
        b.iter(|| slice.sum_masked2(&mask1, &mask2))
    });
}

fn bench_bitslice_sum4_masked2(c: &mut Criterion) {
    let slice = BitSlice::random(NVALUES, 4);
    let mask1 = BitSet::random(NVALUES, 0.25);
    let mask2 = BitSet::random(NVALUES, 0.50);
    c.bench_function("bitslice_sum_masked2_4", move |b| {
        b.iter(|| slice.sum_masked2(&mask1, &mask2))
    });
}

criterion_group!(benches,
                 bench_bitslice_sum1,
                 bench_bitslice_sum2,
                 bench_bitslice_sum4,
                 bench_bitslice_sum1_masked,
                 bench_bitslice_sum2_masked,
                 bench_bitslice_sum4_masked,
                 bench_bitslice_sum1_masked2,
                 bench_bitslice_sum2_masked2,
                 bench_bitslice_sum4_masked2,
                 );
criterion_main!(benches);
