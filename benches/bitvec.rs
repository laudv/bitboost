#[macro_use]
extern crate criterion;
extern crate rand;
extern crate spdyboost;

use criterion::{Criterion, Bencher, Fun};

use spdyboost::bits::BitVec;

const NVALUES: usize = 1_000_000;

fn bench_bitvec_popcnt(b: &mut Bencher, frac1: &f64) {
    let set = BitVec::random(NVALUES, *frac1);
    b.iter(|| set.count_ones_popcnt())
}

fn bench_bitvec_avx2(b: &mut Bencher, frac1: &f64) {
    let set = BitVec::random(NVALUES, *frac1);
    b.iter(|| set.count_ones_avx2())
}

fn bench_bitvec_count_ones(c: &mut Criterion) {
    let popcnt = Fun::new("count_ones_popcnt", bench_bitvec_popcnt);
    let avx2 = Fun::new("count_ones_avx2", bench_bitvec_avx2);
    let funs = vec![popcnt, avx2];

    c.bench_functions("bitvec_count_ones", funs, 0.25f64);
}

fn bench_bitvec_and(c: &mut Criterion) {
    let vec1 = BitVec::random(NVALUES, 0.5);
    let vec2 = BitVec::random(NVALUES, 0.5);
    c.bench_function("bitvec_and", move |b| {
        b.iter(|| vec1.and(&vec2))
    });
}
fn bench_bitvec_andnot(c: &mut Criterion) {
    let vec1 = BitVec::random(NVALUES, 0.5);
    let vec2 = BitVec::random(NVALUES, 0.5);
    c.bench_function("bitvec_andnot", move |b| {
        b.iter(|| vec1.andnot(&vec2))
    });
}


criterion_group!(benches,
                 bench_bitvec_count_ones,
                 bench_bitvec_and,
                 bench_bitvec_andnot,
                 );
criterion_main!(benches);
