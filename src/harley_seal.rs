#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::bitblock::BitBlock;


/// Carry-safe adder.
macro_rules! csa {
    ( $h:ident, $l:ident; $a:ident, $b:ident, $c:ident ) => {{
        let u = _mm256_xor_si256($a, $b);
        $h = _mm256_or_si256(_mm256_and_si256($a, $b), _mm256_and_si256(u, $c));
        $l = _mm256_xor_si256(u, $c);
    }};
    ( $h:ident, $l:ident <- $d1:expr, $d2:expr ) => {{
        let dd1 = $d1;
        let dd2 = $d2;
        csa!($h, $l; $l, dd1, dd2);
    }};
}


// ------------------------------------------------------------------------------------------------

/// Internals that depend on the width of the individual units.
unsafe trait HarleySealInternals {
    type Weights;

    #[inline(always)]
    unsafe fn count_ones(v: __m256i) -> __m256i;

    #[inline(always)]
    unsafe fn vshiftl(a: __m256i, imm8: i32) -> __m256i;

    #[inline(always)]
    unsafe fn vadd(a: __m256i, b: __m256i) -> __m256i;

    #[inline(always)]
    unsafe fn reduce_total(total: __m256i, ws: Self::Weights) -> u64;
}

pub type HarleySealWeights32 = (u8, u8, u8, u8, u8, u8, u8, u8);
struct Internals32;
unsafe impl HarleySealInternals for Internals32 {
    type Weights = HarleySealWeights32;

    #[inline(always)]
    unsafe fn count_ones(v: __m256i) -> __m256i {
        let mut buffer = BitBlock::zeros(); // 64 byte aligned!
        let mut counts = BitBlock::zeros();
        _mm256_store_si256(buffer.as_mut_ptr() as *mut __m256i, v);
        {
            let bufptr = buffer.as_ptr() as *const u32;
            let cntptr = counts.as_mut_ptr() as *mut u32;
            *cntptr.add(0) = (*bufptr.add(0)).count_ones() as u32;
            *cntptr.add(1) = (*bufptr.add(1)).count_ones() as u32;
            *cntptr.add(2) = (*bufptr.add(2)).count_ones() as u32;
            *cntptr.add(3) = (*bufptr.add(3)).count_ones() as u32;
            *cntptr.add(4) = (*bufptr.add(4)).count_ones() as u32;
            *cntptr.add(5) = (*bufptr.add(5)).count_ones() as u32;
            *cntptr.add(6) = (*bufptr.add(6)).count_ones() as u32;
            *cntptr.add(7) = (*bufptr.add(7)).count_ones() as u32;
        };
        _mm256_load_si256(counts.as_ptr() as *const __m256i)
    }

    #[inline(always)]
    unsafe fn vshiftl(a: __m256i, imm8: i32) -> __m256i {
        _mm256_slli_epi32(a, imm8)
    }

    #[inline(always)]
    unsafe fn vadd(a: __m256i, b: __m256i) -> __m256i {
        _mm256_add_epi32(a, b)
    }

    #[inline(always)]
    unsafe fn reduce_total(total: __m256i, ws: Self::Weights) -> u64 {
        let mut sum = 0;
        let x0 = _mm256_extract_epi64(total, 0) as u64;
        let (y0, y1) = (x0 & 0xFFFFFFFF, x0 >> 32); sum += ws.0 as u64 * y0 + ws.1 as u64 * y1;
        let x1 = _mm256_extract_epi64(total, 1) as u64;
        let (y2, y3) = (x1 & 0xFFFFFFFF, x1 >> 32); sum += ws.2 as u64 * y2 + ws.3 as u64 * y3;
        let x2 = _mm256_extract_epi64(total, 2) as u64;
        let (y4, y5) = (x2 & 0xFFFFFFFF, x2 >> 32); sum += ws.4 as u64 * y4 + ws.5 as u64 * y5;
        let x3 = _mm256_extract_epi64(total, 3) as u64;
        let (y6, y7) = (x3 & 0xFFFFFFFF, x3 >> 32); sum += ws.6 as u64 * y6 + ws.7 as u64 * y7;
        sum
    }
}

pub type HarleySealWeights64 = (u8, u8, u8, u8);
struct Internals64;
unsafe impl HarleySealInternals for Internals64 {
    type Weights = HarleySealWeights64;

    #[inline(always)]
    unsafe fn count_ones(v: __m256i) -> __m256i {
        let mut buffer = BitBlock::zeros(); // 64 byte aligned!
        let mut counts = BitBlock::zeros();
        _mm256_store_si256(buffer.as_mut_ptr() as *mut __m256i, v);
        {
            let bufptr = buffer.as_ptr() as *const u64;
            let cntptr = counts.as_mut_ptr() as *mut u64;
            *cntptr.add(0) = (*bufptr.add(0)).count_ones() as u64;
            *cntptr.add(1) = (*bufptr.add(1)).count_ones() as u64;
            *cntptr.add(2) = (*bufptr.add(2)).count_ones() as u64;
            *cntptr.add(3) = (*bufptr.add(3)).count_ones() as u64;
        };
        _mm256_load_si256(counts.as_ptr() as *const __m256i)
    }

    #[inline(always)]
    unsafe fn vshiftl(a: __m256i, imm8: i32) -> __m256i {
        _mm256_slli_epi64(a, imm8)
    }

    #[inline(always)]
    unsafe fn vadd(a: __m256i, b: __m256i) -> __m256i {
        _mm256_add_epi64(a, b)
    }

    #[inline(always)]
    unsafe fn reduce_total(total: __m256i, ws: Self::Weights) -> u64 {
          ws.0 as u64 * _mm256_extract_epi64(total, 0) as u64
        + ws.1 as u64 * _mm256_extract_epi64(total, 1) as u64
        + ws.2 as u64 * _mm256_extract_epi64(total, 2) as u64
        + ws.3 as u64 * _mm256_extract_epi64(total, 3) as u64
    }
}

unsafe fn harley_seal<Input, LoadFn, Internals>(d: &Input, n: usize, load: LoadFn,
        weights: <Internals as HarleySealInternals>::Weights)
    -> u64
where LoadFn: Fn(&Input, usize) -> __m256i,
      Internals: HarleySealInternals,
{
    let mut total = _mm256_setzero_si256();
    let mut b01   = _mm256_setzero_si256();
    let mut b02   = _mm256_setzero_si256();
    let mut b04   = _mm256_setzero_si256();
    let mut b08   = _mm256_setzero_si256();

    let mut b16;
    let mut b02a;
    let mut b02b;
    let mut b04a;
    let mut b04b;
    let mut b08a;
    let mut b08b;

    let mut i = 0;
    while i < n - (n % 16) {
        csa!(b02a, b01 <- load(d, i   ), load(d, i+1 ));
        csa!(b02b, b01 <- load(d, i+2 ), load(d, i+3 ));
        csa!(b04a, b02 <- b02a, b02b);
        csa!(b02a, b01 <- load(d, i+4 ), load(d, i+5 ));
        csa!(b02b, b01 <- load(d, i+6 ), load(d, i+7 ));
        csa!(b04b, b02 <- b02a, b02b);
        csa!(b08a, b04 <- b04a, b04b);
        csa!(b02a, b01 <- load(d, i+8 ), load(d, i+9 ));
        csa!(b02b, b01 <- load(d, i+10), load(d, i+11));
        csa!(b04a, b02 <- b02a, b02b);
        csa!(b02a, b01 <- load(d, i+12), load(d, i+13));
        csa!(b02b, b01 <- load(d, i+14), load(d, i+15));
        csa!(b04b, b02 <- b02a, b02b);
        csa!(b08b, b04 <- b04a, b04b);
        csa!(b16,  b08 <- b08a, b08b);

        total = Internals::vadd(total, Internals::count_ones(b16));
        i += 16;
    }

    total = Internals::vshiftl(total, 4);
    total = Internals::vadd(total, Internals::vshiftl(Internals::count_ones(b08), 3));
    total = Internals::vadd(total, Internals::vshiftl(Internals::count_ones(b04), 2));
    total = Internals::vadd(total, Internals::vshiftl(Internals::count_ones(b02), 1));
    total = Internals::vadd(total,                    Internals::count_ones(b01)    );

    while i < n {
        total = Internals::vadd(total, Internals::count_ones(load(d, i)));
        i += 1;
    }

    Internals::reduce_total(total, weights)
}

fn harley_seal32<Input, LoadFn>(d: &Input, n: usize, load: LoadFn, weights: HarleySealWeights32)
    -> u64
where LoadFn: Fn(&Input, usize) -> __m256i
{
    unsafe {
        harley_seal::<Input, LoadFn, Internals32>(d, n, load, weights)
    }
}

fn harley_seal64<Input, LoadFn>(d: &Input, n: usize, load: LoadFn, weights: HarleySealWeights64)
    -> u64
where LoadFn: Fn(&Input, usize) -> __m256i
{
    unsafe {
        harley_seal::<Input, LoadFn, Internals64>(d, n, load, weights)
    }
}





// ------------------------------------------------------------------------------------------------
