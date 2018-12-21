#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use bits::{BitBlock, BitVec};

unsafe fn count_ones_u64(v: __m256i) -> __m256i {
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

unsafe fn count_ones_u32(v: __m256i) -> __m256i {
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

// Faster Population Counts Using AVX2 Instructions
// Daniel Lemire, Nathan Kurz and Wojciech Mula
// Harvey Seal's algorithm
// https://github.com/CountOnes/hamming_weight/blob/master/src/avx_harley_seal_hamming_weight.c
macro_rules! harvey_seal_aux {
    ( $data:expr, $nblocks:expr, $load_fn:ident, $count_ones:ident ) => {{
        let d = $data;    // data necessary to perform the summation
        let n = $nblocks; // number of 256bit blocks

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
            csa!(b02a, b01 <- $load_fn(d, i   ), $load_fn(d, i+1 ));
            csa!(b02b, b01 <- $load_fn(d, i+2 ), $load_fn(d, i+3 ));
            csa!(b04a, b02 <- b02a, b02b);
            csa!(b02a, b01 <- $load_fn(d, i+4 ), $load_fn(d, i+5 ));
            csa!(b02b, b01 <- $load_fn(d, i+6 ), $load_fn(d, i+7 ));
            csa!(b04b, b02 <- b02a, b02b);
            csa!(b08a, b04 <- b04a, b04b);
            csa!(b02a, b01 <- $load_fn(d, i+8 ), $load_fn(d, i+9 ));
            csa!(b02b, b01 <- $load_fn(d, i+10), $load_fn(d, i+11));
            csa!(b04a, b02 <- b02a, b02b);
            csa!(b02a, b01 <- $load_fn(d, i+12), $load_fn(d, i+13));
            csa!(b02b, b01 <- $load_fn(d, i+14), $load_fn(d, i+15));
            csa!(b04b, b02 <- b02a, b02b);
            csa!(b08b, b04 <- b04a, b04b);
            csa!(b16,  b08 <- b08a, b08b);

            total = _mm256_add_epi64(total, $count_ones(b16));
            i += 16;
        }

        (d, i, n, total, b01, b02, b04, b08)
    }}
}

macro_rules! harvey_seal_64 {
    ( $data:expr, $nblocks:expr, $load_fn:ident, $reduce_fn:ident ) => {{
        let (d, mut i, n, mut total, b01, b02, b04, b08) = 
            harvey_seal_aux!($data, $nblocks, $load_fn, count_ones_u64);

        total = _mm256_slli_epi64(total, 4);
        total = _mm256_add_epi64(total, _mm256_slli_epi64(count_ones_u64(b08), 3));
        total = _mm256_add_epi64(total, _mm256_slli_epi64(count_ones_u64(b04), 2));
        total = _mm256_add_epi64(total, _mm256_slli_epi64(count_ones_u64(b02), 1));
        total = _mm256_add_epi64(total,                   count_ones_u64(b01)    );

        while i < n {
            total = _mm256_add_epi64(total, count_ones_u64($load_fn(d, i)));
            i += 1;
        }

        $reduce_fn(total)
    }}
}

macro_rules! harvey_seal_32 {
    ( $data:expr, $nblocks:expr, $load_fn:ident, $reduce_fn:ident ) => {{
        let (d, mut i, n, mut total, b01, b02, b04, b08) = 
            harvey_seal_aux!($data, $nblocks, $load_fn, count_ones_u32);

        total = _mm256_slli_epi32(total, 4);
        total = _mm256_add_epi32(total, _mm256_slli_epi32(count_ones_u32(b08), 3));
        total = _mm256_add_epi32(total, _mm256_slli_epi32(count_ones_u32(b04), 2));
        total = _mm256_add_epi32(total, _mm256_slli_epi32(count_ones_u32(b02), 1));
        total = _mm256_add_epi32(total,                   count_ones_u32(b01)    );

        while i < n {
            total = _mm256_add_epi32(total, count_ones_u32($load_fn(d, i)));
            i += 1;
        }

        $reduce_fn(total)
    }}
}





// - Harvey-Seal ------------------------------------------------------------------------------- //

unsafe fn load_unmasked(d: &BitVec, i: usize) -> __m256i {
    _mm256_load_si256(d.get_unchecked(i).as_ptr() as *const __m256i)
}

unsafe fn load_and(d: (&BitVec, &BitVec), i: usize) -> __m256i {
    let block = _mm256_load_si256(d.0.get_unchecked(i).as_ptr() as *const __m256i);
    let mask = _mm256_load_si256(d.1.get_unchecked(i).as_ptr() as *const __m256i);
    _mm256_and_si256(block, mask)
}

unsafe fn load_andnot(d: (&BitVec, &BitVec), i: usize) -> __m256i {
    let block = _mm256_load_si256(d.0.get_unchecked(i).as_ptr() as *const __m256i);
    let mask = _mm256_load_si256(d.1.get_unchecked(i).as_ptr() as *const __m256i);
    _mm256_andnot_si256(mask, block)
}

unsafe fn reduce_64(total: __m256i, w0: u64, w1: u64, w2: u64, w3: u64) -> u64 {
      w0 * _mm256_extract_epi64(total, 0) as u64
    + w1 * _mm256_extract_epi64(total, 1) as u64
    + w2 * _mm256_extract_epi64(total, 2) as u64
    + w3 * _mm256_extract_epi64(total, 3) as u64
}

unsafe fn reduce_32(total: __m256i, w0: u64, w1: u64, w2: u64, w3: u64,
                    w4: u64, w5: u64, w6: u64, w7: u64) -> u64 {
    let x0 = _mm256_extract_epi64(total, 0) as u64;
    let x1 = _mm256_extract_epi64(total, 1) as u64;
    let x2 = _mm256_extract_epi64(total, 2) as u64;
    let x3 = _mm256_extract_epi64(total, 3) as u64;
    let y0 = x0 & 0xFFFFFFFF;
    let y1 = x0 >> 32;
    let y2 = x1 & 0xFFFFFFFF;
    let y3 = x1 >> 32;
    let y4 = x2 & 0xFFFFFFFF;
    let y5 = x2 >> 32;
    let y6 = x3 & 0xFFFFFFFF;
    let y7 = x3 >> 32;
    w0*y0 + w1*y1 + w2*y2 + w3*y3 + w4*y4 + w5*y5 + w6*y6 + w7*y7
}

unsafe fn reduce64_1(total: __m256i) -> u64 { reduce_64(total, 1, 1, 1, 1) }
unsafe fn reduce32_1(total: __m256i) -> u64 { reduce_32(total, 1, 1, 1, 1, 1, 1, 1, 1) }


pub fn bitvec_count_ones(blocks: &BitVec) -> u64 {
    unsafe { harvey_seal_64!(blocks, blocks.len(), load_unmasked, reduce64_1) }
}

pub fn bitvec_count_ones32(blocks: &BitVec) -> u64 {
    unsafe { harvey_seal_32!(blocks, blocks.len(), load_unmasked, reduce32_1) }
}

pub fn bitvec_count_and(blocks: &BitVec, mask: &BitVec) -> u64 {
    debug_assert!(blocks.len() == mask.len());
    unsafe { harvey_seal_64!((blocks, mask), blocks.len(), load_and, reduce64_1) }
}

pub fn bitvec_count_andnot(blocks: &BitVec, mask: &BitVec) -> u64 {
    debug_assert!(blocks.len() == mask.len());
    unsafe { harvey_seal_64!((blocks, mask), blocks.len(), load_andnot, reduce64_1) }
}








// - Logical ----------------------------------------------------------------------------------- //

macro_rules! logic_combine {
    ($combiner:ident, $arg1:expr, $arg2:expr) => {{
        let v = $arg1;
        let w = $arg2;
        let nblocks = usize::min(v.nblocks(), w.nblocks());
        let mut vec: Vec<BitBlock> = Vec::with_capacity(nblocks);

        unsafe { vec.set_len(nblocks); }

        let vp = v.as_ptr() as *const __m256i;
        let wp = w.as_ptr() as *const __m256i;
        let vec_ptr = vec.as_mut_ptr() as *mut __m256i;

        for i in 0..nblocks {
            unsafe { 
                let x = $combiner(*vp.add(i), *wp.add(i));
                _mm256_stream_si256(vec_ptr.add(i), x);
            }
        }
        vec
    }}
}

pub fn and(v: &BitVec, w: &BitVec) -> Vec<BitBlock> {
    logic_combine!(_mm256_and_si256, v, w)
}

pub fn andnot(v: &BitVec, w: &BitVec) -> Vec<BitBlock> {
    logic_combine!(_mm256_andnot_si256, w, v)
}
