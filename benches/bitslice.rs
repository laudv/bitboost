#[macro_use]
extern crate criterion;
extern crate rand;
extern crate spdyboost;

use criterion::Criterion;
use spdyboost::bits::{BitSet, BitSlice};
use spdyboost::bits::ScaledBitSlice;

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
        b.iter(|| slice.sum_masked_and(&mask1, &mask2))
    });
}

fn bench_bitslice_sum2_masked2(c: &mut Criterion) {
    let slice = BitSlice::random(NVALUES, 2);
    let mask1 = BitSet::random(NVALUES, 0.25);
    let mask2 = BitSet::random(NVALUES, 0.50);
    c.bench_function("bitslice_sum_masked2_2", move |b| {
        b.iter(|| slice.sum_masked_and(&mask1, &mask2))
    });
}

fn bench_bitslice_sum4_masked2(c: &mut Criterion) {
    let slice = BitSlice::random(NVALUES, 4);
    let mask1 = BitSet::random(NVALUES, 0.25);
    let mask2 = BitSet::random(NVALUES, 0.50);
    c.bench_function("bitslice_sum_masked2_4", move |b| {
        b.iter(|| slice.sum_masked_and(&mask1, &mask2))
    });
}

fn bench_bitslice_count_sum2(c: &mut Criterion) {
    let slice = ScaledBitSlice::<f32>::random(NVALUES, 4, -1.0, 1.0);
    let mask1 = BitSet::random(NVALUES, 0.25);
    let mask2 = BitSet::random(NVALUES, 0.50);
    c.bench_function("bitslice_count_sum2", move |b| {
        b.iter(|| {
            let count = mask1.count_and(&mask2);
            slice.sum_masked_and(&mask1, &mask2, count);
            slice.sum_masked_andnot(&mask1, &mask2, count);
        })
    });
}

fn bench_bitslice_filtered(c: &mut Criterion) {
    let slice = ScaledBitSlice::<f32>::random(NVALUES, 4, -1.0, 1.0);
    let mask1 = BitSet::random(NVALUES, 0.25);
    let mask2 = BitSet::random(NVALUES, 0.50);
    c.bench_function("bitslice_filtered", move |b| {
        b.iter(|| {
            slice.sum_filtered(&mask1, &mask2);
        })
    });
}

criterion_group!(benches,
                 //bench_bitslice_sum1,
                 //bench_bitslice_sum2,
                 //bench_bitslice_sum4,
                 //bench_bitslice_sum1_masked,
                 //bench_bitslice_sum2_masked,
                 //bench_bitslice_sum4_masked,
                 //bench_bitslice_sum1_masked2,
                 //bench_bitslice_sum2_masked2,
                 //bench_bitslice_sum4_masked2,
                 bench_bitslice_count_sum2,
                 bench_bitslice_filtered,
                 );
criterion_main!(benches);
