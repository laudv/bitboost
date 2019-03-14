#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::bitblock::BitBlock;

// get_grad_sum
//   - uncompressed / compresed
//   - width: 1, 2, 4
//
// components:
//   - bitvec AND count
//   - bitslice masked sum

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

#[allow(dead_code)]
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

#[allow(unused_macros)]
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







// ------------------------------------------------------------------------------------------------

#[allow(unused)]
unsafe fn debug_print(reg: __m256i) {
    let mut buf = [0u64; 4];
    _mm256_store_si256(buf.as_mut_ptr() as *mut __m256i, reg);
    for i in 0..4 {
        println!("{}: {:064b}", i, buf[i]);
        //println!("{}: {}", i, buf[i]);
    }
}

#[allow(unused)]
unsafe fn debug_print32x8(reg: __m256i) {
    let mut buf = [0u32; 8];
    _mm256_store_si256(buf.as_mut_ptr() as *mut __m256i, reg);
    for i in 0..8 {
        //println!("{}: {:032b}", i, buf[i]);
        println!("{}: {}", i, buf[i]);
    }
}

#[allow(unused)]
unsafe fn debug_print32x4(reg: __m128i) {
    let mut buf = [0u32; 4];
    _mm_store_si128(buf.as_mut_ptr() as *mut __m128i, reg);
    for i in 0..4 {
        println!("{}: {:032b}", i, buf[i]);
        //println!("{}: {}", i, buf[i]);
    }
}



unsafe fn load_and(d: (*const __m256i, *const __m256i), i: usize) -> __m256i {
    let block = _mm256_load_si256(d.0.add(i));
    let mask  = _mm256_load_si256(d.1.add(i));
    _mm256_and_si256(block, mask)
}

unsafe fn load_and_c(d: (*const __m256i, *const __m256i, *const i32), i: usize) -> __m256i {
    let block = _mm256_load_si256(d.0.add(i));
    let indices = _mm256_load_si256(d.1.add(i));
    let mask = _mm256_i32gather_epi32(d.2, indices, 4);
    _mm256_and_si256(block, mask)
}

unsafe fn load_mask_and_u32_w1_uc(d: (*const __m256i, *const __m256i, *const __m256i), i: usize)
    -> __m256i
{
    let block = _mm256_load_si256(d.0.add(i));
    let mask1 = _mm256_load_si256(d.1.add(i));
    let mask2 = _mm256_load_si256(d.2.add(i));
    _mm256_and_si256(block, _mm256_and_si256(mask1, mask2))
}

unsafe fn load_mask_and_u32_w2_uc(d: (*const __m256i, *const __m128i, *const __m128i), i: usize)
    -> __m256i
{
    let block = _mm256_load_si256(d.0.add(i));
    let mask128 = _mm_and_si128(_mm_load_si128(d.1.add(i)), _mm_load_si128(d.2.add(i)));
    let mask = _mm256_set_m128i(mask128, mask128);
    _mm256_and_si256(block, mask)
}

unsafe fn load_mask_and_u32_w4_uc(d: (*const __m256i, *const i64, *const i64), i: usize)
    -> __m256i
{
    let block = _mm256_load_si256(d.0.add(i));
    let mask64 = *d.1.add(i) & *d.2.add(i);
    let mask = _mm256_set1_epi64x(mask64);
    _mm256_and_si256(block, mask)
}

unsafe fn load_mask_and_u32_w8_uc(d: (*const __m256i, *const u32, *const u32), i: usize)
    -> __m256i
{
    let block = _mm256_load_si256(d.0.add(i));
    let mask32 = *d.1.add(i) & *d.2.add(i);
    let mask64 = (((mask32 as u64) << 32) | mask32 as u64) as i64;
    let mask = _mm256_set1_epi64x(mask64);
    _mm256_and_si256(block, mask)
}

unsafe fn load_mask_and_u32_w1_c(d: (*const __m256i, *const __m256i, *const __m256i, *const i32),
                                 i: usize) -> __m256i
{
    // d = (bitslice, indices, node example mask, feature value mask)
    let block = _mm256_load_si256(d.0.add(i));
    let indices = _mm256_load_si256(d.1.add(i));
    let mask1 = _mm256_load_si256(d.2.add(i));
    let mask2 = _mm256_i32gather_epi32(d.3, indices, 4);
    _mm256_and_si256(block, _mm256_and_si256(mask1, mask2))
}

unsafe fn load_mask_and_u32_w2_c(d: (*const __m256i, *const __m128i, *const __m128i, *const i32),
                                 i: usize) -> __m256i
{
    let block = _mm256_load_si256(d.0.add(i));
    let indices = _mm_load_si128(d.1.add(i));
    let mask1 = _mm_load_si128(d.2.add(i));
    let mask2 = _mm_i32gather_epi32(d.3, indices, 4);
    let mask128 = _mm_and_si128(mask1, mask2);
    let mask = _mm256_set_m128i(mask128, mask128);
    _mm256_and_si256(block, mask)
}

unsafe fn load_mask_and_u32_w4_c(d: (*const __m256i, *const u32, *const i64, *const u32), i: usize)
    -> __m256i
{
    let block = _mm256_load_si256(d.0.add(i));
    let (index1, index2) = (*d.1.add(2*i) as usize, *d.1.add(2*i+1) as usize);
    let mask1 = *d.2.add(i);
    let mask2 = (*d.3.add(index1) as u64) | ((*d.3.add(index2) as u64) << 32);
    let mask64 = mask1 & mask2 as i64;
    let mask = _mm256_set1_epi64x(mask64);
    _mm256_and_si256(block, mask)
}

unsafe fn load_mask_and_u32_w8_c(d: (*const __m256i, *const u32, *const u32, *const u32), i: usize)
    -> __m256i
{
    let block = _mm256_load_si256(d.0.add(i));
    let index = *d.1.add(i) as usize;
    let mask1 = *d.2.add(i);
    let mask2 = *d.3.add(index);
    let mask32 = mask1 & mask2;
    let mask64 = (((mask32 as u64) << 32) | mask32 as u64) as i64;
    let mask = _mm256_set1_epi64x(mask64);
    _mm256_and_si256(block, mask)
}

unsafe fn reduce_64(total: __m256i, w0: u64, w1: u64, w2: u64, w3: u64) -> u64 {
      w0 * _mm256_extract_epi64(total, 0) as u64
    + w1 * _mm256_extract_epi64(total, 1) as u64
    + w2 * _mm256_extract_epi64(total, 2) as u64
    + w3 * _mm256_extract_epi64(total, 3) as u64
}

unsafe fn reduce_32(total: __m256i, w0: u64, w1: u64, w2: u64, w3: u64,
                    w4: u64, w5: u64, w6: u64, w7: u64) -> u64 {
    let mut sum = 0;
    let x0 = _mm256_extract_epi64(total, 0) as u64;
    let (y0, y1) = (x0 & 0xFFFFFFFF, x0 >> 32); sum += w0*y0 + w1*y1;
    let x1 = _mm256_extract_epi64(total, 1) as u64;
    let (y2, y3) = (x1 & 0xFFFFFFFF, x1 >> 32); sum += w2*y2 + w3*y3;
    let x2 = _mm256_extract_epi64(total, 2) as u64;
    let (y4, y5) = (x2 & 0xFFFFFFFF, x2 >> 32); sum += w4*y4 + w5*y5;
    let x3 = _mm256_extract_epi64(total, 3) as u64;
    let (y6, y7) = (x3 & 0xFFFFFFFF, x3 >> 32); sum += w6*y6 + w7*y7;
    sum
}

unsafe fn reduce64_1(total: __m256i) -> u64 { reduce_64(total, 1, 1, 1, 1) }
unsafe fn reduce64_2(total: __m256i) -> u64 { reduce_64(total, 1, 1, 2, 2) }
unsafe fn reduce64_4(total: __m256i) -> u64 { reduce_64(total, 1, 2, 4, 8) }
unsafe fn reduce32_8(total: __m256i) -> u64 { reduce_32(total, 1, 2, 4, 8, 16, 32, 64, 128) }





// ------------------------------------------------------------------------------------------------

// btslce -> bitslice
// summx -> sum masked, width=x
// uc   -> uncompressed
// c    -> compressed
// nm   -> node mask (node examples mask)
// fm   -> feature mask (if compressed, uses the 'compressed' indices)

pub unsafe fn bitvec_count_and_uc(v1: &[BitBlock], v2: &[BitBlock]) -> u64 {
    let nblocks = usize::min(v1.len(), v2.len());
    let ptr1 = v1.as_ptr() as *const __m256i;
    let ptr2 = v2.as_ptr() as *const __m256i;
    harvey_seal_64!((ptr1, ptr2), nblocks, load_and, reduce64_1)
}

pub unsafe fn bitvec_count_and_c(v1: &[BitBlock], idxs: &[BitBlock], v2: &[BitBlock]) -> u64 {
    let nblocks = usize::min(v1.len(), idxs.len());
    let ptr1 = v1.as_ptr() as *const __m256i;
    let ptr2 = idxs.as_ptr() as *const __m256i;
    let ptr3 = v2.as_ptr() as *const i32;
    harvey_seal_64!((ptr1, ptr2, ptr3), nblocks, load_and_c, reduce64_1)
}

pub unsafe fn btslce_summ1_uc(slice: &[BitBlock], nm: &[BitBlock], fm: &[BitBlock]) -> u64 {
    let nblocks = slice.len();
    debug_assert_eq!(nm.len(), nblocks);
    debug_assert_eq!(fm.len(), nblocks);

    let ptr1 = slice.as_ptr() as *const __m256i;
    let ptr2 = nm.as_ptr() as *const __m256i;
    let ptr3 = fm.as_ptr() as *const __m256i;

    harvey_seal_64!((ptr1, ptr2, ptr3), nblocks, load_mask_and_u32_w1_uc, reduce64_1)
}

pub unsafe fn btslce_summ2_uc(slice: &[BitBlock], nm: &[BitBlock], fm: &[BitBlock]) -> u64 {
    let nblocks = slice.len();
    debug_assert_eq!(nm.len() * 2, nblocks);
    debug_assert_eq!(fm.len() * 2, nblocks);

    let ptr1 = slice.as_ptr() as *const __m256i;
    let ptr2 = nm.as_ptr() as *const __m128i;
    let ptr3 = fm.as_ptr() as *const __m128i;

    harvey_seal_64!((ptr1, ptr2, ptr3), nblocks, load_mask_and_u32_w2_uc, reduce64_2)
}

pub unsafe fn btslce_summ4_uc(slice: &[BitBlock], nm: &[BitBlock], fm: &[BitBlock]) -> u64 {
    let nblocks = slice.len();
    debug_assert_eq!(nm.len() * 4, nblocks);
    debug_assert_eq!(fm.len() * 4, nblocks);

    let ptr1 = slice.as_ptr() as *const __m256i;
    let ptr2 = nm.as_ptr() as *const i64;
    let ptr3 = fm.as_ptr() as *const i64;

    harvey_seal_64!((ptr1, ptr2, ptr3), nblocks, load_mask_and_u32_w4_uc, reduce64_4)
}

pub unsafe fn btslce_summ8_uc(slice: &[BitBlock], nm: &[BitBlock], fm: &[BitBlock]) -> u64 {
    let nblocks = slice.len();
    debug_assert_eq!(nm.len() * 8, nblocks);
    debug_assert_eq!(fm.len() * 8, nblocks);

    let ptr1 = slice.as_ptr() as *const __m256i;
    let ptr2 = nm.as_ptr() as *const u32;
    let ptr3 = fm.as_ptr() as *const u32;

    harvey_seal_32!((ptr1, ptr2, ptr3), nblocks, load_mask_and_u32_w8_uc, reduce32_8)
}

pub unsafe fn btslce_summ1_c(slice: &[BitBlock], indices: &[BitBlock], nm: &[BitBlock],
                             fm: &[BitBlock]) -> u64
{
    let nblocks = slice.len();
    debug_assert_eq!(indices.len(), nblocks);
    debug_assert_eq!(nm.len(), nblocks);

    let ptr1 = slice.as_ptr() as *const __m256i;
    let ptr2 = indices.as_ptr() as *const __m256i;
    let ptr3 = nm.as_ptr() as *const __m256i;
    let ptr4 = fm.as_ptr() as *const i32;

    harvey_seal_64!((ptr1, ptr2, ptr3, ptr4), nblocks, load_mask_and_u32_w1_c, reduce64_1)
}

pub unsafe fn btslce_summ2_c(slice: &[BitBlock], indices: &[BitBlock], nm: &[BitBlock],
                             fm: &[BitBlock]) -> u64
{
    let nblocks = slice.len();
    debug_assert_eq!(indices.len() * 2, nblocks);
    debug_assert_eq!(nm.len() * 2, nblocks);

    let ptr1 = slice.as_ptr() as *const __m256i;
    let ptr2 = indices.as_ptr() as *const __m128i;
    let ptr3 = nm.as_ptr() as *const __m128i;
    let ptr4 = fm.as_ptr() as *const i32;

    harvey_seal_64!((ptr1, ptr2, ptr3, ptr4), nblocks, load_mask_and_u32_w2_c, reduce64_2)
}

pub unsafe fn btslce_summ4_c(slice: &[BitBlock], indices: &[BitBlock], nm: &[BitBlock],
                             fm: &[BitBlock]) -> u64
{
    let nblocks = slice.len();
    debug_assert_eq!(indices.len() * 4, nblocks);
    debug_assert_eq!(nm.len() * 4, nblocks);

    let ptr1 = slice.as_ptr() as *const __m256i;
    let ptr2 = indices.as_ptr() as *const u32;
    let ptr3 = nm.as_ptr() as *const i64;
    let ptr4 = fm.as_ptr() as *const u32;

    harvey_seal_64!((ptr1, ptr2, ptr3, ptr4), nblocks, load_mask_and_u32_w4_c, reduce64_4)
}

pub unsafe fn btslce_summ8_c(slice: &[BitBlock], indices: &[BitBlock], nm: &[BitBlock],
                             fm: &[BitBlock]) -> u64
{
    let nblocks = slice.len();
    debug_assert_eq!(indices.len() * 8, nblocks);
    debug_assert_eq!(nm.len() * 8, nblocks);

    let ptr1 = slice.as_ptr() as *const __m256i;
    let ptr2 = indices.as_ptr() as *const u32;
    let ptr3 = nm.as_ptr() as *const u32;
    let ptr4 = fm.as_ptr() as *const u32;

    harvey_seal_32!((ptr1, ptr2, ptr3, ptr4), nblocks, load_mask_and_u32_w8_c, reduce32_8)
}





// ------------------------------------------------------------------------------------------------

pub unsafe fn or_assign(bv0: &mut [BitBlock], bv1: &[BitBlock]) {
    debug_assert_eq!(bv0.len(), bv1.len());
    for (b0, b1) in bv0.iter_mut().zip(bv1.iter()) {
        let p0 = b0.as_mut_ptr() as *mut __m256i;
        let p1 = b1.as_ptr() as *const __m256i;
        *p0 = _mm256_or_si256(*p0, *p1);
    }
}
