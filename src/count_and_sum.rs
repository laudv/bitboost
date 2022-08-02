/*
 * Copyright 2019 DTAI Research Group - KU Leuven.
 * License: Apache License 2.0
 * Author: Laurens Devos
*/

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::bitblock::{BitBlock, BitBlocks};
use crate::bitslice::{BitsliceLayout, BitsliceWithLayout};
use crate::bitslice::{BitsliceLayout1, BitsliceLayout2, BitsliceLayout4, BitsliceLayout8};


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

    unsafe fn count_ones(v: __m256i) -> __m256i;
    unsafe fn vshiftl<const IMM8: i32>(a: __m256i) -> __m256i;
    unsafe fn vadd(a: __m256i, b: __m256i) -> __m256i;
    unsafe fn reduce_total(total: __m256i, ws: Self::Weights) -> u64;
}

type HarleySealWeights32 = (u8, u8, u8, u8, u8, u8, u8, u8);
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
    unsafe fn vshiftl<const IMM8: i32>(a: __m256i) -> __m256i {
        _mm256_slli_epi32(a, IMM8)
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

type HarleySealWeights64 = (u8, u8, u8, u8);
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
    unsafe fn vshiftl<const IMM8: i32>(a: __m256i) -> __m256i {
        _mm256_slli_epi64(a, IMM8)
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

    total = Internals::vshiftl::<4>(total);
    total = Internals::vadd(total, Internals::vshiftl::<3>(Internals::count_ones(b08)));
    total = Internals::vadd(total, Internals::vshiftl::<2>(Internals::count_ones(b04)));
    total = Internals::vadd(total, Internals::vshiftl::<1>(Internals::count_ones(b02)));
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



#[allow(unused_macros)]
macro_rules! loadfn {
    ($name:ident < Uncompressed > () ) => {
        fn $name(&(bset1,          bset2):
                     &(*const __m256i, *const __m256i),
                     index: usize)
            -> __m256i
        {
            unsafe { 
                let mask1 = loadfn!(@load __m256i: bset1, index);
                let mask2 = loadfn!(@load __m256i: bset2, index);
                _mm256_and_si256(mask1, mask2)
            }
        }
    };

    ($name:ident < Compressed10 > () ) => {
        fn $name(&(indexes1,       bset1,          bset2):
                     &(*const __m256i, *const __m256i, *const __m256i),
                     index: usize) 
            -> __m256i
        {
            unsafe { 
                let mask1 = loadfn!(@load   __m256i: bset1, index);
                let mask2 = loadfn!(@gather __m256i: indexes1, bset2, index);
                _mm256_and_si256(mask1, mask2)
            }
        }
    };

    ($name:ident < Uncompressed > ( $in:ident ) ) => {
        fn $name(&(bitslice,       bset1,          bset2):
                     &(*const __m256i, *const __m256i, *const __m256i),
                     index: usize)
            -> __m256i
        {
            unsafe {
                let slice = loadfn!(@load __m256i: bitslice, index);
                let mask1 = loadfn!(@load     $in: bset1, index);
                let mask2 = loadfn!(@load     $in: bset2, index);
                let maskx = loadfn!(@and      $in: mask1, mask2);
                let maskx = loadfn!(@expand   $in: maskx);
                _mm256_and_si256(slice, maskx)
            }
        }
    };

    ($name:ident < Compressed10 > ( $in:ident ) ) => {
        fn $name(&(bitslice,       indexes1,       bset1,          bset2):
                 &(*const __m256i, *const __m256i, *const __m256i, *const __m256i),
                 index: usize)
            -> __m256i
        {
            unsafe {
                let slice = loadfn!(@load __m256i: bitslice, index);
                let mask1 = loadfn!(@load     $in: bset1, index);
                let mask2 = loadfn!(@gather   $in: indexes1, bset2, index);
                let maskx = loadfn!(@and      $in: mask1, mask2);
                let maskx = loadfn!(@expand   $in: maskx);
                _mm256_and_si256(slice, maskx)
            }
        }
    };

    ($name:ident < Compressed01 > ( $in:ident ) ) => {
        fn $name(&(bitslice,       bset1,          indexes2,       bset2):
                 &(*const __m256i, *const __m256i, *const __m256i, *const __m256i),
                 index: usize)
            -> __m256i
        {
            unsafe {
                let slice = loadfn!(@select $in: indexes2, bitslice, index);
                let mask1 = loadfn!(@gather $in: indexes2, bset1, index);
                let mask2 = loadfn!(@load   $in: bset2, index);
                let maskx = loadfn!(@and    $in: mask1, mask2);
                let maskx = loadfn!(@expand $in: maskx);
                _mm256_and_si256(slice, maskx)
            }
        }
    };

    // 8-bit discretization
    (@load   u32: $bset:expr, $index:expr) => {{ *($bset as *const u32).add($index) }};
    (@and    u32: $m1:expr, $m2:expr) => {{ $m1 & $m2 }};
    (@expand u32: $value:expr) => {{
        let v32 = $value;
        let v64 = (((v32 as u64) << 32) | v32 as u64) as i64;
        _mm256_set1_epi64x(v64)
    }};
    (@gather u32: $indexes:ident, $bset:expr, $index:expr) => {{
        let index = *(($indexes as *const u32).add($index)) as usize;
        *($bset as *const u32).add(index)
    }};
    (@select u32: $indexes:ident, $slice:expr, $index:expr) => {{
        let index = *(($indexes as *const u32).add($index)) as usize;
        _mm256_load_si256($slice.add(index))
    }};

    // 4-bit discretization
    (@load   u64: $bset:expr, $index:expr) => {{ *($bset as *const u64).add($index) }};
    (@and    u64: $v1:expr, $v2:expr) => {{ $v1 & $v2 }};
    (@expand u64: $value:expr) => {{ let v64 = $value; _mm256_set1_epi64x(v64 as i64) }};
    (@gather u64: $indexes:ident, $bset:expr, $index:expr) => {{
        let index1 = *(($indexes as *const u32).add(2*$index  )) as usize;
        let index2 = *(($indexes as *const u32).add(2*$index+1)) as usize;
        // 1  1   2  2   3  3   4  4
        // m1 m2  m1 m2  m1 m2  m1 m2
        //
        // m1 = 32-bit mask in bset at index `index1`
        // m2 = 32-bit mask in bset at index `index2`
        let bset = $bset as *const u32;
        (*bset.add(index1) as u64) | ((*bset.add(index2) as u64) << 32)
    }};
    (@select u64: $indexes:ident, $slice:expr, $index:expr) => {{
        let index = *(($indexes as *const u32).add(2 * $index)) as usize;
        _mm256_load_si256($slice.add(index / 2))
    }};

    // 2-bit discretization
    (@load   __m128i: $bset:expr, $index:expr) => {{
        _mm_load_si128(($bset as *const __m128i).add($index)) }};
    (@and    __m128i: $v1:expr, $v2:expr) => {{ _mm_and_si128($v1, $v2) }};
    (@expand __m128i: $value:expr) => {{ let v128 = $value; _mm256_set_m128i(v128, v128) }};
    (@gather __m128i: $indexes:ident, $bset:expr, $index:expr) => {{
        let indexes = _mm_load_si128(($indexes as *const __m128i).add($index));
        _mm_i32gather_epi32($bset as *const i32, indexes, 4)
    }};
    (@select __m128i: $indexes:ident, $slice:expr, $index:expr) => {{
        let index = *(($indexes as *const u32).add(4 * $index)) as usize;
        _mm256_load_si256($slice.add(index / 4))
    }};

    // 1-bit discretization
    (@load   __m256i: $bset:expr, $index:expr) => {{
        _mm256_load_si256(($bset as *const __m256i).add($index)) }};
    (@and    __m256i: $v1:expr, $v2:expr) => {{ _mm256_and_si256($v1, $v2) }};
    (@expand __m256i: $value:expr) => {{ $value }};
    (@gather __m256i: $indexes:ident, $bset:expr, $index:expr) => {{
        let indexes = _mm256_load_si256(($indexes as *const __m256i).add($index));
        _mm256_i32gather_epi32($bset as *const i32, indexes, 4)
    }};
    (@select __m256i: $indexes:ident, $slice:expr, $index:expr) => {{
        let index = *(($indexes as *const u32).add(8 * $index)) as usize;
        _mm256_load_si256($slice.add(index / 8))
    }}
}


loadfn!(load_and2_c00<Uncompressed>());
loadfn!(load_and2_c10<Compressed10>());

loadfn!(load_and3_w1_c00<Uncompressed>(__m256i));
loadfn!(load_and3_w2_c00<Uncompressed>(__m128i));
loadfn!(load_and3_w4_c00<Uncompressed>(u64));
loadfn!(load_and3_w8_c00<Uncompressed>(u32));

loadfn!(load_and3_w1_c10<Compressed10>(__m256i));
loadfn!(load_and3_w2_c10<Compressed10>(__m128i));
loadfn!(load_and3_w4_c10<Compressed10>(u64));
loadfn!(load_and3_w8_c10<Compressed10>(u32));

loadfn!(load_and3_w1_c01<Compressed01>(__m256i));
loadfn!(load_and3_w2_c01<Compressed01>(__m128i));
loadfn!(load_and3_w4_c01<Compressed01>(u64));
loadfn!(load_and3_w8_c01<Compressed01>(u32));






// - Harley-Seal based counting and summing: uncompressed, compressed10,and compressed01 ----------

pub fn count_and2_c00(blocks1: &BitBlocks, blocks2: &BitBlocks) -> u64 {
    let ptr1 = blocks1.as_ptr() as *const __m256i;
    let ptr2 = blocks2.as_ptr() as *const __m256i;
    harley_seal64(&(ptr1, ptr2), blocks1.len(), load_and2_c00, (1, 1, 1, 1))
}

pub fn count_and2_c10(indexes1: &BitBlocks, blocks1: &BitBlocks, blocks2: &BitBlocks) -> u64 {
    let idxs = indexes1.as_ptr() as *const __m256i;
    let ptr1 = blocks1.as_ptr() as *const __m256i;
    let ptr2 = blocks2.as_ptr() as *const __m256i;
    harley_seal64(&(idxs, ptr1, ptr2), blocks1.len(), load_and2_c10, (1, 1, 1, 1))
}

pub fn sum_and3_w1_c00(bitslice: &BitBlocks, bset1: &BitBlocks, bset2: &BitBlocks) -> u64 {
    let slice = bitslice.as_ptr() as *const __m256i;
    let ptr1 = bset1.as_ptr() as *const __m256i;
    let ptr2 = bset2.as_ptr() as *const __m256i;
    harley_seal64(&(slice, ptr1, ptr2), bitslice.len(), load_and3_w1_c00, (1, 1, 1, 1))
}

pub fn sum_and3_w2_c00(bitslice: &BitBlocks, bset1: &BitBlocks, bset2: &BitBlocks) -> u64 {
    let slice = bitslice.as_ptr() as *const __m256i;
    let ptr1 = bset1.as_ptr() as *const __m256i;
    let ptr2 = bset2.as_ptr() as *const __m256i;
    harley_seal64(&(slice, ptr1, ptr2), bitslice.len(), load_and3_w2_c00, (1, 1, 2, 2))
}

pub fn sum_and3_w4_c00(bitslice: &BitBlocks, bset1: &BitBlocks, bset2: &BitBlocks) -> u64 {
    let slice = bitslice.as_ptr() as *const __m256i;
    let ptr1 = bset1.as_ptr() as *const __m256i;
    let ptr2 = bset2.as_ptr() as *const __m256i;
    harley_seal64(&(slice, ptr1, ptr2), bitslice.len(), load_and3_w4_c00, (1, 2, 4, 8))
}

pub fn sum_and3_w8_c00(bitslice: &BitBlocks, bset1: &BitBlocks, bset2: &BitBlocks) -> u64 {
    let slice = bitslice.as_ptr() as *const __m256i;
    let ptr1 = bset1.as_ptr() as *const __m256i;
    let ptr2 = bset2.as_ptr() as *const __m256i;
    harley_seal32(&(slice, ptr1, ptr2), bitslice.len(), load_and3_w8_c00,
                  (1, 2, 4, 8, 16, 32, 64, 128))
}

pub fn sum_and3_w1_c10(bitslice: &BitBlocks, idxs1: &BitBlocks, bset1: &BitBlocks,
                      bset2: &BitBlocks) -> u64 {
    let slice = bitslice.as_ptr() as *const __m256i;
    let idxs = idxs1.as_ptr() as *const __m256i;
    let ptr1 = bset1.as_ptr() as *const __m256i;
    let ptr2 = bset2.as_ptr() as *const __m256i;
    harley_seal64(&(slice, idxs, ptr1, ptr2), bitslice.len(), load_and3_w1_c10, (1, 1, 1, 1))
}

pub fn sum_and3_w2_c10(bitslice: &BitBlocks, idxs1: &BitBlocks, bset1: &BitBlocks,
                      bset2: &BitBlocks) -> u64 {
    let slice = bitslice.as_ptr() as *const __m256i;
    let idxs = idxs1.as_ptr() as *const __m256i;
    let ptr1 = bset1.as_ptr() as *const __m256i;
    let ptr2 = bset2.as_ptr() as *const __m256i;
    harley_seal64(&(slice, idxs, ptr1, ptr2), bitslice.len(), load_and3_w2_c10, (1, 1, 2, 2))
}

pub fn sum_and3_w4_c10(bitslice: &BitBlocks, idxs1: &BitBlocks, bset1: &BitBlocks,
                      bset2: &BitBlocks) -> u64 {
    let slice = bitslice.as_ptr() as *const __m256i;
    let idxs = idxs1.as_ptr() as *const __m256i;
    let ptr1 = bset1.as_ptr() as *const __m256i;
    let ptr2 = bset2.as_ptr() as *const __m256i;
    harley_seal64(&(slice, idxs, ptr1, ptr2), bitslice.len(), load_and3_w4_c10, (1, 2, 4, 8))
}

pub fn sum_and3_w8_c10(bitslice: &BitBlocks, idxs1: &BitBlocks, bset1: &BitBlocks,
                      bset2: &BitBlocks) -> u64 {
    let slice = bitslice.as_ptr() as *const __m256i;
    let idxs = idxs1.as_ptr() as *const __m256i;
    let ptr1 = bset1.as_ptr() as *const __m256i;
    let ptr2 = bset2.as_ptr() as *const __m256i;
    harley_seal32(&(slice, idxs, ptr1, ptr2), bitslice.len(), load_and3_w8_c10,
                  (1, 2, 4, 8, 16, 32, 64, 128))
}







// - Non-Harley-Seal based counting: compressed11; does not fit well in Harley-Seal template ------

macro_rules! count_and_sum_c11 {
    (count_and2_c11) => {
        pub fn count_and2_c11(idxs1: &BitBlocks, bset1: &BitBlocks,
                             idxs2: &BitBlocks, bset2: &BitBlocks) -> u64 {
            count_and_sum_c11!(@body count: idxs1, bset1, idxs2, bset2, 1, ())
        }
    };

    (sum_and3_wx_c11: $name:ident, $width:expr, $layout:ident) => {
        pub fn $name(bitslice: &BitBlocks,
                     idxs1: &BitBlocks, bset1: &BitBlocks,
                     idxs2: &BitBlocks, bset2: &BitBlocks) -> u64 {
            count_and_sum_c11!(@body sum: idxs1, bset1, idxs2, bset2, $width,
                               BitsliceWithLayout::<_, $layout>::for_bitblocks(bitslice))
        }
    };

    (@body $count_or_sum:ident: $idxs1:expr, $bset1:expr, $idxs2:expr, $bset2:expr,
           $width:expr, $bitslice:expr) => {{
        let mut i1 = 0;
        let mut i2 = 0;
        let mut sums = [0u64; $width];
        let idxs1 = $idxs1.cast::<u32>();
        let idxs2 = $idxs2.cast::<u32>();
        let bset1 = $bset1.cast::<u32>();
        let bset2 = $bset2.cast::<u32>();

        #[allow(unused_variables)]
        let bitslice = $bitslice;

        safety_check!(idxs1.len() == bset1.len());
        safety_check!(idxs2.len() == bset2.len());

        while i1 < idxs1.len() && i2 < idxs2.len() {
            let idx1 = unsafe { idxs1.get_unchecked(i1) };
            let idx2 = unsafe { idxs2.get_unchecked(i2) };

            if idx1 < idx2      { i1 += 1; }
            else if idx1 > idx2 { i2 += 1; }
            else { // equal!
                let m1 = unsafe { bset1.get_unchecked(i1) };
                let m2 = unsafe { bset2.get_unchecked(i2) };
                let mask = m1 & m2;
                for lane in 0..$width {
                    let sum = unsafe { sums.get_unchecked_mut(lane) };
                    let block = count_and_sum_c11!(@load $count_or_sum: bitslice, mask, i1, lane);
                    *sum += block.count_ones() as u64;
                }
                i1 += 1;
                i2 += 1;
            }
        }

        let mut sum = 0;
        for lane in 0..$width {
            sum += (1<<lane) * sums[lane];
        }
        sum
    }};

    (@load sum:   $bitslice:expr, $mask:expr, $index:expr, $lane:expr) => {{
        let block = unsafe { $bitslice.get_block_unchecked($index, $lane) };
        block & $mask
    }};

    (@load count: $bitslice:expr, $mask:expr, $index:expr, $lane:expr) => {{
        $mask
    }}
}

count_and_sum_c11!(count_and2_c11);
count_and_sum_c11!(sum_and3_wx_c11: sum_and3_w1_c11, 1, BitsliceLayout1);
count_and_sum_c11!(sum_and3_wx_c11: sum_and3_w2_c11, 2, BitsliceLayout2);
count_and_sum_c11!(sum_and3_wx_c11: sum_and3_w4_c11, 4, BitsliceLayout4);
count_and_sum_c11!(sum_and3_wx_c11: sum_and3_w8_c11, 8, BitsliceLayout8);










// ------------------------------------------------------------------------------------------------



#[cfg(test)]
mod test {

    use crate::bitblock::{BitBlocks};
    use crate::bitset::Bitset;
    use crate::bitslice::{Bitslice, BitsliceLayout};
    use crate::bitslice::{BitsliceLayout1, BitsliceLayout2, BitsliceLayout4, BitsliceLayout8};

    use super::*;

    #[test]
    fn test_count_and2_c00() {
        let n = 20_000;
        let seed = 112;
        let mut sum = 0;
        let mut bset1 = Bitset::zeros(n);
        let mut bset2 = Bitset::zeros(n);

        for i in 0..n {
            let b1 = (((1123*i) % 3 + 51) % 3 + seed) % 3 == 1;
            let b2 = (((8827*i) % 2 + 99) % 2 + seed) % 2 == 1;
            bset1.set_bit(i, b1);
            bset2.set_bit(i, b2);
            if b1 && b2 { sum += 1; }
        }

        let bset1_ptr = bset1.as_ptr() as *const __m256i;
        let bset2_ptr = bset2.as_ptr() as *const __m256i;
        let sum_harley_seal1 = harley_seal64(&(bset1_ptr, bset2_ptr), bset1.len(),
                                             load_and2_c00, (1, 1, 1, 1));
        let sum_harley_seal2 = count_and2_c00(&bset1, &bset2);
        assert_eq!(sum, sum_harley_seal1);
        assert_eq!(sum, sum_harley_seal2);
    }

    #[test]
    fn test_count_and2_c10_1() {
        let m1 = 4*128;
        let m2 = 4*1024;
        let seed = 112;
        let mut sum = 0;
        let mut bset1 = Bitset::zeros(m1 * 32);
        let mut bset2 = Bitset::zeros(m2 * 32);
        let indexes = BitBlocks::from_iter(m1, 0u32..m1 as u32);

        for i in 0..(32*m1) {
            let b1 = (((1123*i) % 3 + 51) % 3 + seed) % 3 == 1;
            let b2 = (((8827*i) % 2 + 99) % 2 + seed) % 2 == 1;
            bset1.set_bit(i, b1);
            bset2.set_bit(i, b2);
            if b1 && b2 { sum += 1; }
        }

        let indexes_ptr = indexes.as_ptr() as *const __m256i;
        let bset1_ptr = bset1.as_ptr() as *const __m256i;
        let bset2_ptr = bset2.as_ptr() as *const __m256i;
        let sum_harley_seal1 = harley_seal64(&(indexes_ptr, bset1_ptr, bset2_ptr),
                                             bset1.len(), load_and2_c10, (1, 1, 1, 1));
        let sum_harley_seal2 = count_and2_c10(&indexes, &bset1, &bset2);
        assert_eq!(sum, sum_harley_seal1);
        assert_eq!(sum, sum_harley_seal2);
    }

    #[test]
    fn test_count_and2_c10_2() {
        let m1 = 4*128;
        let m2 = 4*1024;
        let seed = 92;
        let mut sum = 0;
        let mut bset1 = Bitset::zeros(m1 * 32);
        let mut bset2 = Bitset::zeros(m2 * 32);
        let indexes_iter = (0..m1).map(|i| ((((i*31)%m2)+12284+seed)%m2) as u32);
        let mut indexes = BitBlocks::from_iter(m1, indexes_iter);
        indexes.cast_mut::<u32>()[0..m1].sort();

        for i in 0..m1 {
            for j in 0..32 {
                let b1 = (((1123*(i*32+j)) % 3 + 51) % 3 + seed) % 3 == 1;
                let b2 = (((8827*(i*32+j)) % 2 + 99) % 2 + seed) % 2 == 1;
                bset1.set_bit(i*32+j, b1);
                bset2.set_bit(*indexes.get::<u32>(i) as usize*32+j, b2);
                if b1 && b2 { sum += 1; }
            }
        }

        let indexes_ptr = indexes.as_ptr() as *const __m256i;
        let bset1_ptr = bset1.as_ptr() as *const __m256i;
        let bset2_ptr = bset2.as_ptr() as *const __m256i;
        let sum_harley_seal1 = harley_seal64(&(indexes_ptr, bset1_ptr, bset2_ptr),
                                             bset1.len(), load_and2_c10, (1, 1, 1, 1));
        let sum_harley_seal2 = count_and2_c10(&indexes, &bset1, &bset2);
        assert_eq!(sum, sum_harley_seal1);
        assert_eq!(sum, sum_harley_seal2);
    }

    #[test]
    fn test_count_and2_c11() {
    }

    #[test]
    fn test_sum_and3_c00() {
        sum_c00::<BitsliceLayout1>(20);
        sum_c00::<BitsliceLayout2>(111);
        sum_c00::<BitsliceLayout4>(222);
        sum_c00::<BitsliceLayout8>(333);
    }

    fn sum_c00<L>(seed: usize)
    where L: BitsliceLayout
    {
        let n = 20_000;
        let m = L::nunique_values();

        let mut bitslice = Bitslice::new();
        let mut view = bitslice.with_layout_mut::<L>();
        view.resize(n);

        let nblocks = view.nblocks();

        let mut bset1 = Bitset::zeros(n);
        let mut bset2 = Bitset::zeros(n);
        let mut sum = 0;

        for i in 0..n {
            let k = ((((1123*i) % m + 51) % m + seed) % m) as u8;
            let b1 = (((1123*i) % 3 + 51) % 3 + seed) % 3 == 1;
            let b2 = (((8827*i) % 2 + 99) % 2 + seed) % 2 == 1;
            view.set_value(i, k);
            bset1.set_bit(i, b1);
            bset2.set_bit(i, b2);
            if b1 && b2 {
                sum += k as u64;
                //println!("{:3}: {:5} = {}", i, k, sum);
            }
        }

        let bitslice_ptr = bitslice.as_bitblocks().as_ptr() as *const __m256i;
        let bset1_ptr = bset1.as_ptr() as *const __m256i;
        let bset2_ptr = bset2.as_ptr() as *const __m256i;

        let w1 = (1, 1, 1, 1);
        let w2 = (1, 1, 2, 2);
        let w4 = (1, 2, 4, 8);
        let w8 = (1, 2, 4, 8, 16, 32, 64, 128);

        let sum_harley_seal1 = match L::width() {
            1 => { harley_seal64(&(bitslice_ptr, bset1_ptr, bset2_ptr), nblocks, load_and3_w1_c00, w1) }
            2 => { harley_seal64(&(bitslice_ptr, bset1_ptr, bset2_ptr), nblocks, load_and3_w2_c00, w2) }
            4 => { harley_seal64(&(bitslice_ptr, bset1_ptr, bset2_ptr), nblocks, load_and3_w4_c00, w4) }
            8 => { harley_seal32(&(bitslice_ptr, bset1_ptr, bset2_ptr), nblocks, load_and3_w8_c00, w8) }
            _ => { panic!() }
        };

        let sum_harley_seal2 = match L::width() {
            1 => { sum_and3_w1_c00(bitslice.as_bitblocks(), &bset1, &bset2) }
            2 => { sum_and3_w2_c00(bitslice.as_bitblocks(), &bset1, &bset2) }
            4 => { sum_and3_w4_c00(bitslice.as_bitblocks(), &bset1, &bset2) }
            8 => { sum_and3_w8_c00(bitslice.as_bitblocks(), &bset1, &bset2) }
            _ => { panic!() }
        };

        assert_eq!(sum, sum_harley_seal1);
        assert_eq!(sum, sum_harley_seal2);
    }

    #[test]
    fn test_sum_and3_c10() {
        sum_c10::<BitsliceLayout1>(20);
        sum_c10::<BitsliceLayout2>(111);
        sum_c10::<BitsliceLayout4>(222);
        sum_c10::<BitsliceLayout8>(333);
    }

    fn sum_c10<L>(seed: usize)
    where L: BitsliceLayout
    {
        let m1 = 2048;
        let m2 = 4*m1;
        let m = L::nunique_values();

        let mut bitslice = Bitslice::new();
        let mut view = bitslice.with_layout_mut::<L>();
        view.resize(m1 * 32);

        let nblocks = view.nblocks();

        let mut bset1 = Bitset::zeros(m1 * 32);
        let mut bset2 = Bitset::zeros(m2 * 32);
        let mut sum = 0;

        let indexes_iter = (0..m1).map(|i| ((((i*79)%m2)+12284+seed)%m2) as u32);
        let mut indexes = BitBlocks::from_iter(m1, indexes_iter);
        indexes.cast_mut::<u32>()[0..m1].sort();

        for i in 0..m1 {
            for j in 0..32 {
                let k = ((((1123*i+j) % m + 51) % m + seed) % m) as u8;
                let b1 = (((1123*i+j) % 3 + 51) % 3 + seed) % 3 == 1;
                let b2 = (((8827*i+j) % 2 + 99) % 2 + seed) % 2 == 1;
                view.set_value(i*32+j, k);
                bset1.set_bit(i*32+j, b1);
                bset2.set_bit(*indexes.get::<u32>(i) as usize*32+j, b2);
                if b1 && b2 {
                    sum += k as u64;
                    //println!("{:3}: {:5} = {}", i, k, sum);
                }
            }
        }

        let bitslice_ptr = bitslice.as_bitblocks().as_ptr() as *const __m256i;
        let indexes1_ptr = indexes.as_ptr() as *const __m256i;
        let bset1_ptr = bset1.as_ptr() as *const __m256i;
        let bset2_ptr = bset2.as_ptr() as *const __m256i;

        let w1 = (1, 1, 1, 1);
        let w2 = (1, 1, 2, 2);
        let w4 = (1, 2, 4, 8);
        let w8 = (1, 2, 4, 8, 16, 32, 64, 128);

        let sum_harley_seal1 = match L::width() {
            1 => { harley_seal64(&(bitslice_ptr, indexes1_ptr, bset1_ptr, bset2_ptr), nblocks, load_and3_w1_c10, w1) }
            2 => { harley_seal64(&(bitslice_ptr, indexes1_ptr, bset1_ptr, bset2_ptr), nblocks, load_and3_w2_c10, w2) }
            4 => { harley_seal64(&(bitslice_ptr, indexes1_ptr, bset1_ptr, bset2_ptr), nblocks, load_and3_w4_c10, w4) }
            8 => { harley_seal32(&(bitslice_ptr, indexes1_ptr, bset1_ptr, bset2_ptr), nblocks, load_and3_w8_c10, w8) }
            _ => { panic!() }
        };

        let sum_harley_seal2 = match L::width() {
            1 => { sum_and3_w1_c10(bitslice.as_bitblocks(), &indexes, &bset1, &bset2) }
            2 => { sum_and3_w2_c10(bitslice.as_bitblocks(), &indexes, &bset1, &bset2) }
            4 => { sum_and3_w4_c10(bitslice.as_bitblocks(), &indexes, &bset1, &bset2) }
            8 => { sum_and3_w8_c10(bitslice.as_bitblocks(), &indexes, &bset1, &bset2) }
            _ => { panic!() }
        };

        assert_eq!(sum, sum_harley_seal1);
        assert_eq!(sum, sum_harley_seal2);
    }

    #[test]
    fn test_sum_and3_c11() {
        sum_c11::<BitsliceLayout1>();
        sum_c11::<BitsliceLayout2>();
        sum_c11::<BitsliceLayout4>();
        sum_c11::<BitsliceLayout8>();
    }

    fn sum_c11<L>()
    where L: BitsliceLayout
    {
        let m = L::nunique_values();

        let mut bset1 = Bitset::zeros(3 * 32);
        bset1.enable_bit(0);
        bset1.enable_bit(2);
        bset1.enable_bit(31);
        bset1.enable_bit(64);
        bset1.enable_bit(65);
        bset1.enable_bit(66);

        let mut bset2 = Bitset::zeros(3 * 32);
        bset2.enable_bit(32);
        bset2.enable_bit(33);
        bset2.enable_bit(34);
        bset2.enable_bit(35);
        bset2.enable_bit(63);
        bset2.enable_bit(65);
        bset2.enable_bit(96);

        let indexes1 = BitBlocks::from_iter::<u32, _>(3, [1u32, 2, 12].iter().cloned());
        let indexes2 = BitBlocks::from_iter::<u32, _>(4, [0u32, 1, 12, 14].iter().cloned());

        let mut bitslice = Bitslice::new();
        let mut view = bitslice.with_layout_mut::<L>();
        view.resize(3 * 32);
        view.set_value( 0, (7  % m) as u8);
        view.set_value( 1, (11 % m) as u8);
        view.set_value( 2, (13 % m) as u8);
        view.set_value( 3, (15 % m) as u8);
        view.set_value(31, (97 % m) as u8);
        view.set_value(32, (23 % m) as u8);
        view.set_value(63, (27 % m) as u8);
        view.set_value(64, (31 % m) as u8);
        view.set_value(65, (33 % m) as u8);
        view.set_value(66, (37 % m) as u8);
        view.set_value(67, (51 % m) as u8);
        view.set_value(96, (51 % m) as u8);

        let sum_check = 
              view.get_value(0 ) as u64
            + view.get_value(2 ) as u64
            + view.get_value(31) as u64
            + view.get_value(65) as u64;

        println!("indexes1 {:?}", &indexes1.cast::<u32>()[0..3]);
        println!("indexes2 {:?}", &indexes2.cast::<u32>()[0..4]);

        let sum = match L::width() {
            1 => { sum_and3_w1_c11(bitslice.as_bitblocks(), &indexes1, &bset1, &indexes2, &bset2) }
            2 => { sum_and3_w2_c11(bitslice.as_bitblocks(), &indexes1, &bset1, &indexes2, &bset2) }
            4 => { sum_and3_w4_c11(bitslice.as_bitblocks(), &indexes1, &bset1, &indexes2, &bset2) }
            8 => { sum_and3_w8_c11(bitslice.as_bitblocks(), &indexes1, &bset1, &indexes2, &bset2) }
            _ => { panic!() }
        };

        println!("sum {}, check {}", sum, sum_check);
        assert_eq!(sum, sum_check);
    }
}
