#[macro_use]
extern crate criterion;
extern crate rand;
extern crate spdyboost;

use criterion::Criterion;
use spdyboost::bits::{ValueStore, BitSlice4};

mod util;

const NVALUES: usize = 1_000_000;

fn bench_bitslice2(c: &mut Criterion) {
    c.bench_function("bitslice2_sum", |b| {
        let slice = util::gen_bitslice2(NVALUES);
        let select = util::gen_select_full(NVALUES, 0.5);
        b.iter(|| slice.sum_full(&select))
    });
}

fn bench_bitslice4(c: &mut Criterion) {
    c.bench_function("bitslice4_sum", |b| {
        let slice = util::gen_bitslice4(NVALUES);
        let select = util::gen_select_full(NVALUES, 0.5);
        b.iter(|| slice.sum_full(&select))
    });
}

criterion_group!(benches,
                 //bench_bitslice2,
                 bench_bitslice4,
                 );
criterion_main!(benches);
