#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use std::fmt::{Debug, Result as FmtResult, Formatter};
use std::mem::transmute;

use bits::bitblock::{BitBlockOps, BitBlock};
use bits::{Mask, ValueStore};
use bits::{FullBitSet};

fn get_block_index_and_bitpos(bit_index: usize) -> (usize, usize) {
    let block_index = bit_index / BitBlock::nbits();
    let bitpos = bit_index % BitBlock::nbits();
    (block_index, bitpos)
}


// --------------------------------------------------------------------------------------------- //

unsafe fn count(v: __m256i) -> __m256i {
    // Rustc generates fast code :-)
    let mut buf: [u64; 4] = [0; 4];
    let mut counts: [u64; 4] = [0; 4];
    _mm256_storeu_si256(buf[..].as_mut_ptr() as *mut __m256i, v);
    for i in 0..4 { counts[i] = buf[i].count_ones() as u64; }
    _mm256_loadu_si256(counts[..].as_mut_ptr() as *mut __m256i)
}

/// load 4 times 64 bits of data blocks `[lsb, lsb+1, lsb+2, msb]` and a single mask block of 64
/// bits. Return 256 bits of AND'ed data blocks `[lsb & mask, lsb+1 & mask, lsb+2 & mask, msb &
/// mask ]`.
#[inline(always)]
unsafe fn load4<M: Mask>(v: &[BitBlock], m: &M, i: &mut usize) -> __m256i {
    let w = v.as_ptr() as *const __m256i;
    let k = m.get_index_unchecked(*i);
    let b = m.get_block_unchecked(*i);
    *i += 1;

    let b256 = _mm256_set1_epi64x(transmute::<BitBlock, i64>(b));
    _mm256_and_si256(_mm256_loadu_si256(w.add(k)), b256)
}

#[inline(always)]
unsafe fn load2<M: Mask>(v: &[BitBlock], m: &M, i: &mut usize) -> __m256i { 
    let w = v.as_ptr() as *const __m128i;
    let k1 = m.get_index_unchecked(*i);
    let b1 = m.get_block_unchecked(*i);
    *i += 1;
    let k2 = m.get_index_unchecked(*i);
    let b2 = m.get_block_unchecked(*i);
    *i += 1;

    let buf: [u64; 4] = [b1, b1, b2, b2];
    let b256 = _mm256_loadu_si256(buf[..].as_ptr() as *const __m256i);
    let d256 = _mm256_set_m128i(_mm_loadu_si128(w.add(k1)), _mm_loadu_si128(w.add(k2)));

    _mm256_and_si256(d256, b256)
}

#[allow(dead_code)]
#[inline(never)]
unsafe fn print_values64(reg: __m256i, bits: bool) {
    let mut stack: [u64; 4] = [0; 4];
    _mm256_storeu_si256(stack[..].as_mut_ptr() as *mut __m256i, reg);
    if bits { for i in 0..4 { print!("{:064b} ", stack[i]); } }
    else    { for i in 0..4 { print!("{:12} ", stack[i]); } }
    println!();
}

// 3-bit carry save adder
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
macro_rules! sum_avx2 {
    ( $self:ident, $mask:ident, $load:ident, [ $w0:expr, $w1:expr, $w2:expr, $w3:expr ] ) => {{
        // TODO sanity check -> move to function
        //debug_assert_eq!($mask.nblocks(), $self.blocks.len() / $width);

        let v = &$self.blocks;
        let n = $mask.nstored();

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
            csa!(b02a, b01 <- $load(v, $mask, &mut i), $load(v, $mask, &mut i));
            csa!(b02b, b01 <- $load(v, $mask, &mut i), $load(v, $mask, &mut i));
            csa!(b04a, b02 <- b02a, b02b);
            csa!(b02a, b01 <- $load(v, $mask, &mut i), $load(v, $mask, &mut i));
            csa!(b02b, b01 <- $load(v, $mask, &mut i), $load(v, $mask, &mut i));
            csa!(b04b, b02 <- b02a, b02b);
            csa!(b08a, b04 <- b04a, b04b);
            csa!(b02a, b01 <- $load(v, $mask, &mut i), $load(v, $mask, &mut i));
            csa!(b02b, b01 <- $load(v, $mask, &mut i), $load(v, $mask, &mut i));
            csa!(b04a, b02 <- b02a, b02b);
            csa!(b02a, b01 <- $load(v, $mask, &mut i), $load(v, $mask, &mut i));
            csa!(b02b, b01 <- $load(v, $mask, &mut i), $load(v, $mask, &mut i));
            csa!(b04b, b02 <- b02a, b02b);
            csa!(b08b, b04 <- b04a, b04b);
            csa!(b16,  b08 <- b08a, b08b);
            total = _mm256_add_epi64(total, count(b16));
        }

        total = _mm256_slli_epi64(total, 4);
        total = _mm256_add_epi64(total, _mm256_slli_epi64(count(b08), 3));
        total = _mm256_add_epi64(total, _mm256_slli_epi64(count(b04), 2));
        total = _mm256_add_epi64(total, _mm256_slli_epi64(count(b02), 1));
        total = _mm256_add_epi64(total,                   count(b01)    );

        while i < n {
            total = _mm256_add_epi64(total, count($load(v, $mask, &mut i)));
        }

          $w0 * _mm256_extract_epi64(total, 0) as u64
        + $w1 * _mm256_extract_epi64(total, 1) as u64
        + $w2 * _mm256_extract_epi64(total, 2) as u64
        + $w3 * _mm256_extract_epi64(total, 3) as u64
    }}
}





// --------------------------------------------------------------------------------------------- //



macro_rules! generate_bitslice {
    ($name:ident, $width:expr) => {
        pub struct $name {
            blocks: Vec<BitBlock>,
        }

        impl $name {
            pub fn new(nbits: usize) -> $name {
                let nblocks = BitBlock::nblocks(nbits);
                $name {
                    blocks: vec![0; nblocks * $width],
                }
            }
        }

        impl ValueStore for $name {
            type Value = u64;

            fn nvalues(&self) -> usize {
                (self.blocks.len() / $width) * BitBlock::nbits()
            }

            fn get_value(&self, index: usize) -> u64 {
                let mut res: BitBlock = 0;
                let (i, j) = get_block_index_and_bitpos(index);
                for k in 0..$width {
                    let block = self.blocks[i * $width + k];
                    let value = block.get_bit(j);
                    res = res.set_bit(k, value);
                }
                res as u64
            }

            fn set_value(&mut self, index: usize, value: u64) {
                let (i, j) = get_block_index_and_bitpos(index);
                let bits = value as BitBlock;
                for k in 0..$width {
                    let block = &mut self.blocks[i * $width + k];
                    let value = bits.get_bit(k);
                    *block = block.set_bit(j, value);
                }

            }

            fn sum_full(&self, mask: &FullBitSet) -> u64 {
                unsafe { self.sum_full_unsafe(mask) }
            }
        }

        impl Debug for $name {
            fn fmt(&self, f: &mut Formatter) -> FmtResult {
                writeln!(f, "{}[{}×{}]", stringify!($name), $width, self.nvalues())?;
                for i in 0..self.blocks.len() / $width {
                    for k in 0..$width {
                        let block = self.blocks[i * $width + k];
                        writeln!(f, "{:3}.{:1}: {:064b}", i, k, block)?;
                    }
                    writeln!(f)?;
                }
                Ok(())
            }
        }
    }
}

generate_bitslice!(BitSlice2, 2);
generate_bitslice!(BitSlice4, 4);

// TODO macrofy
impl BitSlice2 {
    unsafe fn sum_full_unsafe(&self, mask: &FullBitSet) -> u64 {
        sum_avx2!(self, mask, load2, [1, 2, 1, 2])
    }
}

// TODO macrofy
impl BitSlice4 {
    unsafe fn sum_full_unsafe(&self, mask: &FullBitSet) -> u64 {
        sum_avx2!(self, mask, load4, [1, 2, 4, 8])
    }
}








// --------------------------------------------------------------------------------------------- //

#[cfg(test)]
mod test {
    use super::{BitSlice2, BitSlice4};
    use bits::ValueStore;
    use bits::FullBitSet;

    fn test_bitslice_basic1<M: ValueStore<Value = u64>>(k: usize, mut store: M) {
        for i in 0..(1<<k) { store.set_value(i, i as M::Value); }
        for i in 0..(1<<k) {
            let value = store.get_value(i);
            assert_eq!(i as M::Value, value);
        }
    }

    fn test_bitslice_basic2<M: ValueStore<Value = u64>>(n: usize, k: usize, mut store: M) {
        let mask = (1<<k)-1;
        let rand = |i| (((4321 + i * 1234) % 1619) % mask) as M::Value;
        for i in 0..n { store.set_value(i, rand(i)); }
        for i in 0..n {
            let (c, v) = (rand(i), store.get_value(i));
            assert_eq!(c, v);
        }
    }

    fn test_bitslice_sum<M: ValueStore<Value = u64> + std::fmt::Debug>(n: usize, k: usize, mut store: M, f: fn(usize) -> bool) {
        let mask = (1<<k)-1;
        let rand = |i| (((4321 + i * 1234) % 1619) % mask) as M::Value;
        let bs = FullBitSet::from_bool_iter((0..n).map(|i| f(i)));
        let mut sum0 = 0;
        for i in 0..n {
            let val = rand(i);
            if f(i) { sum0 += val; print!("+{}", val); }
            store.set_value(i, val);
        }
        println!();
        println!("store: {:?}", store);
        let sum1 = store.sum_full(&bs);
        println!("sum0 {}, sum1 {}", sum0, sum1);
        assert_eq!(sum0, sum1);
    }

    #[test]
    fn test_bitslice1() {
        test_bitslice_basic1(2, BitSlice2::new(1<<2));
        test_bitslice_basic1(4, BitSlice4::new(1<<4));
    }

    #[test]
    fn test_bitslice2() {
        let n = 10_000;
        test_bitslice_basic2(n, 2, BitSlice2::new(n));
        test_bitslice_basic2(n, 4, BitSlice4::new(n));
    }

    #[test]
    fn test_sum1() {
        let n = 1024;
        test_bitslice_sum(n, 4, BitSlice4::new(n), |i| i%7==1);
        test_bitslice_sum(n, 4, BitSlice4::new(n), |i| i%131==1);
        test_bitslice_sum(n, 2, BitSlice2::new(n), |i| i%7==1);
        test_bitslice_sum(n, 2, BitSlice2::new(n), |i| i%131==1);
        //panic!();
    }
}
