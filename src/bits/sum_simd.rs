#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use bits::{BitBlock, BitSet, BitVec};

unsafe fn count_ones_u64(v: __m256i) -> __m256i {
    // -- IMPL 1 --
    //let mut buf: [u64; 4] = [0; 4];
    //let mut counts: [u64; 4] = [0; 4];
    //_mm256_storeu_si256(buf[..].as_mut_ptr() as *mut __m256i, v);
    //for i in 0..4 { counts[i] = buf[i].count_ones() as u64; }
    //_mm256_loadu_si256(counts[..].as_mut_ptr() as *mut __m256i)

    // -- IMPL 2 --
    let mut buffer = BitBlock::zero(); // 32 byte aligned!
    let mut counts = BitBlock::zero();
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

    // -- IMPL 3 --
    //_mm256_set_epi64x(
    //    _mm256_extract_epi64(v, 0).count_ones() as i64,
    //    _mm256_extract_epi64(v, 1).count_ones() as i64,
    //    _mm256_extract_epi64(v, 2).count_ones() as i64,
    //    _mm256_extract_epi64(v, 3).count_ones() as i64,
    //)
}

// Load a 256bit register from memory without any masks applied to it.
unsafe fn load_unmasked(v: &BitVec, i: usize) -> __m256i {
    _mm256_load_si256(v.get_unchecked(i).as_ptr() as *const __m256i)
}

unsafe fn load_masked1(v: (&BitVec, &BitSet), i: usize) -> __m256i {
    let block = _mm256_load_si256(v.0.get_unchecked(i).as_ptr() as *const __m256i);
    let mask = _mm256_load_si256(v.1.get_unchecked(i).as_ptr() as *const __m256i);
    _mm256_and_si256(block, mask)
}

unsafe fn load_masked2(v: (&BitVec, &BitSet), i: usize) -> __m256i {
    let block = _mm256_load_si256(v.0.get_unchecked(i).as_ptr() as *const __m256i);
    let mask128 = _mm_load_si128((v.1.as_ptr() as *const __m128i).add(i));
    let mask = _mm256_permute4x64_epi64(_mm256_broadcastsi128_si256(mask128), 0b01011010);
    _mm256_and_si256(block, mask)
}

unsafe fn load_masked4(v: (&BitVec, &BitSet), i: usize) -> __m256i {
    let block = _mm256_load_si256(v.0.get_unchecked(i).as_ptr() as *const __m256i);
    let mask64 = *(v.1.as_ptr() as *const i64).add(i);
    let mask = _mm256_set1_epi64x(mask64);
    _mm256_and_si256(block, mask)
}

unsafe fn generic_reduce(total: __m256i, w0: u64, w1: u64, w2: u64, w3: u64) -> u64 {
      w0 * _mm256_extract_epi64(total, 0) as u64
    + w1 * _mm256_extract_epi64(total, 1) as u64
    + w2 * _mm256_extract_epi64(total, 2) as u64
    + w3 * _mm256_extract_epi64(total, 3) as u64
}

unsafe fn reduce1(total: __m256i) -> u64 { generic_reduce(total, 1, 1, 1, 1) }
unsafe fn reduce2_pow2(total: __m256i) -> u64 { generic_reduce(total, 1, 2, 1, 2) }
unsafe fn reduce4_pow2(total: __m256i) -> u64 { generic_reduce(total, 1, 2, 4, 8) }

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
macro_rules! harvey_seal_avx2 {
    ( $data:expr, $nblocks:expr, $load_fn:ident, $reduce_fn:ident ) => {{
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

            total = _mm256_add_epi64(total, count_ones_u64(b16));
            i += 16;
        }

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






// - Access functions -------------------------------------------------------------------------- //

pub fn bitset_count_ones(blocks: &BitVec) -> u64 {
    unsafe { harvey_seal_avx2!(blocks, blocks.len(), load_unmasked, reduce1) }
}

pub fn bitslice_sum_width1(blocks: &BitVec) -> u64 {
    unsafe { harvey_seal_avx2!(blocks, blocks.len(), load_unmasked, reduce1) }
}

pub fn bitslice_sum_width2(blocks: &BitVec) -> u64 {
    unsafe { harvey_seal_avx2!(blocks, blocks.len(), load_unmasked, reduce2_pow2) }
}

pub fn bitslice_sum_width4(blocks: &BitVec) -> u64 {
    unsafe { harvey_seal_avx2!(blocks, blocks.len(), load_unmasked, reduce4_pow2) }
}

pub fn bitslice_sum_masked_width1(blocks: &BitVec, mask: &BitSet) -> u64 {
    unsafe { harvey_seal_avx2!((blocks, mask), blocks.len(), load_masked1, reduce1) }
}

pub fn bitslice_sum_masked_width2(blocks: &BitVec, mask: &BitSet) -> u64 {
    unsafe { harvey_seal_avx2!((blocks, mask), blocks.len(), load_masked2, reduce2_pow2) }
}

pub fn bitslice_sum_masked_width4(blocks: &BitVec, mask: &BitSet) -> u64 {
    unsafe { harvey_seal_avx2!((blocks, mask), blocks.len(), load_masked4, reduce4_pow2) }
}
