#[macro_use]
extern crate criterion;
extern crate rand;
extern crate spdyboost;

use criterion::{Criterion};

use spdyboost::bits::BitVec;

const NVALUES: usize = 1_000_000;

fn bench_bitvec_count_ones(c: &mut Criterion) {
    c.bench_function_over_inputs("bitvec_count_ones", move |b, &sz| {
        let vec1 = BitVec::random(sz, 0.5);
        b.iter(|| vec1.count_ones())
    }, vec![NVALUES]);
}
fn bench_bitvec_count_ones32(c: &mut Criterion) {
    c.bench_function_over_inputs("bitvec_count_ones32", move |b, &sz| {
        let vec1 = BitVec::random(sz, 0.5);
        b.iter(|| vec1.count_ones32())
    }, vec![NVALUES]);
}
fn bench_bitvec_count_and(c: &mut Criterion) {
    let vec1 = BitVec::random(NVALUES, 0.5);
    let vec2 = BitVec::random(NVALUES, 0.5);
    c.bench_function("bitvec_count_and", move |b| {
        b.iter(|| vec1.count_and(&vec2))
    });
}
fn bench_bitvec_count_andnot(c: &mut Criterion) {
    let vec1 = BitVec::random(NVALUES, 0.5);
    let vec2 = BitVec::random(NVALUES, 0.5);
    c.bench_function("bitvec_count_andnot", move |b| {
        b.iter(|| vec1.count_andnot(&vec2))
    });
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
                 bench_bitvec_count_ones32,
                 //bench_bitvec_count_and,
                 //bench_bitvec_count_andnot,
                 //bench_bitvec_and,
                 //bench_bitvec_andnot,
                 );
criterion_main!(benches);
