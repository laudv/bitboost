use bits::{BitBlock, BitVec, BitSet};
use bits::bitblock::{get_bit, set_bit, get_blockpos, get_bitpos};
use bits::simd;

static SUM_FUN_PTRS: [fn(&BitVec) -> u64; 4] = [
    simd::bitslice_sum_width1,
    simd::bitslice_sum_width2,
    simd::bitslice_sum_width2, // invalid
    simd::bitslice_sum_width4,
];

static SUM_MASKED_FUN_PTRS: [fn(&BitVec, &BitVec) -> u64; 4] = [
    simd::bitslice_sum_masked_width1,
    simd::bitslice_sum_masked_width2,
    simd::bitslice_sum_masked_width2, // invalid
    simd::bitslice_sum_masked_width4,
];

static SUM_MASKED_AND_FUN_PTRS: [fn(&BitVec, &BitVec, &BitVec) -> u64; 4] = [
    simd::bitslice_sum_masked_and_width1,
    simd::bitslice_sum_masked_and_width2,
    simd::bitslice_sum_masked_and_width2, // invalid
    simd::bitslice_sum_masked_and_width4,
];

static SUM_MASKED_ANDNOT_FUN_PTRS: [fn(&BitVec, &BitVec, &BitVec) -> u64; 4] = [
    simd::bitslice_sum_masked_andnot_width1,
    simd::bitslice_sum_masked_andnot_width2,
    simd::bitslice_sum_masked_andnot_width2, // invalid
    simd::bitslice_sum_masked_andnot_width4,
];

pub struct BitSlice {
    vec: BitVec,
    width: u8,
}

impl BitSlice {
    pub fn new(nbits: usize, width: u8) -> BitSlice {
        assert!(width == 1 || width == 2 || width == 4, "width {} not supported", width);

        let nblocks = BitBlock::blocks_required_for(nbits);
        let vec = BitVec::zero_blocks(nblocks * width as usize);

        BitSlice {
            vec: vec,
            width: width,
        }
    }

    pub fn random(nbits: usize, width: u8) -> BitSlice {
        use rand::prelude::*;
        use rand::distributions::Uniform;

        let mask = (1 << width) - 1;
        let mut rng = thread_rng();
        let dist = Uniform::new(0u8, 0x23);
        let mut slice = BitSlice::new(nbits, width);

        for i in 0..nbits {
            slice.set_value(i, dist.sample(&mut rng) & mask);
        }

        slice
    }

    pub fn get_value(&self, index: usize) -> u8 {
        let mut res = 0;
        let i = get_blockpos::<u64>(index);
        let j = get_bitpos::<u64>(index);
        let vec_u32 = self.vec.cast::<u64>();
        for k in 0u8..self.width {
            let bits = vec_u32[i * self.width as usize + k as usize];
            let b = get_bit(bits, j);
            res = set_bit(res, k, b);
        }
        res
    }

    pub fn set_value(&mut self, index: usize, value: u8) {
        let i = get_blockpos::<u64>(index);
        let j = get_bitpos::<u64>(index);
        let vec_u32 = self.vec.cast_mut::<u64>();
        for k in 0u8..self.width {
            let bits = &mut vec_u32[i * self.width as usize + k as usize];
            let b = get_bit(value, k);
            *bits = set_bit(*bits, j, b);
        }
    }

    pub fn nblocks(&self) -> usize { self.vec.nblocks() }
    pub fn nbytes(&self) -> usize { self.vec.nbytes() }
    pub fn nbits(&self) -> usize { self.vec.nbits() }
    pub fn width(&self) -> u8 { self.width }
    pub fn nunique_values(&self) -> u8 { 1 << self.width }

    pub fn sum(&self) -> u64 {
        debug_assert!(self.width == 1 || self.width == 2 || self.width == 4);
        let f = unsafe { SUM_FUN_PTRS.get_unchecked(self.width as usize - 1) };
        f(&self.vec)
    }

    pub fn sum_masked(&self, mask: &BitSet) -> u64 {
        debug_assert!(self.width == 1 || self.width == 2 || self.width == 4);
        let f = unsafe { SUM_MASKED_FUN_PTRS.get_unchecked(self.width as usize - 1) };
        f(&self.vec, mask)
    }

    pub fn sum_masked_and(&self, mask1: &BitSet, mask2: &BitSet) -> u64 {
        debug_assert!(self.width == 1 || self.width == 2 || self.width == 4);
        let f = unsafe { SUM_MASKED_AND_FUN_PTRS.get_unchecked(self.width as usize - 1) };
        f(&self.vec, mask1, mask2)
    }

    pub fn sum_masked_andnot(&self, mask1: &BitSet, mask2: &BitSet) -> u64 {
        debug_assert!(self.width == 1 || self.width == 2 || self.width == 4);
        let f = unsafe { SUM_MASKED_ANDNOT_FUN_PTRS.get_unchecked(self.width as usize - 1) };
        f(&self.vec, mask1, mask2)
    }
}



#[cfg(test)]
mod test {
    use bits::{BitSlice, BitVec};

    #[test]
    fn test_bitslice_len() {
        let bs = BitSlice::new(100, 1u8);
        assert_eq!(bs.nblocks(), 1);

        let bs = BitSlice::new(100, 2u8);
        assert_eq!(bs.nblocks(), 2);

        let bs = BitSlice::new(257, 2u8);
        assert_eq!(bs.nblocks(), 4);
    }

    #[test]
    fn test_bitslice_set_value() {
        let n = 10_000;
        for &width in &[1u8, 2u8, 4u8] {
            let mut bs = BitSlice::new(n, width);
            for i in 0..n {
                let v = ((i*7+13) % (1<<width)) as u8;
                bs.set_value(i, v);
                assert_eq!(bs.get_value(i), v);
            }
            for i in n..bs.nbits() / (width as usize) {
                assert_eq!(bs.get_value(i), 0);
            }
        }
    }

    #[test] #[should_panic]
    fn test_bitslice_wrong_width3() { BitSlice::new(10, 3); }

    #[test] #[should_panic]
    fn test_bitslice_wrong_width5() { BitSlice::new(10, 5); }

    #[test]
    fn test_bitslice_sum() {
        let n = 10_000;
        for &width in &[1u8, 2u8, 4u8] {
            let mut bs = BitSlice::new(n, width);
            assert_eq!(bs.sum(), 0);
            let mut sum1: u64 = 0;
            for i in 0..n {
                let v = ((i*7+13) % (1<<width)) as u8;
                bs.set_value(i, v);
                sum1 += v as u64;
            }
            let sum2 = bs.sum();
            println!("width {}: sums: {} / {}", width, sum1, sum2);
            assert_eq!(sum1, sum2);
        }

    }

    #[test]
    fn test_bitslice_sum_masked() {
        let n = 10_000;
        for &width in &[1u8, 2u8, 4u8] {
            let mut bs = BitSlice::new(n, width);
            let mut mask = BitVec::zero_bits(n);

            assert_eq!(bs.sum(), 0);

            let mut sum1: u64 = 0;
            for i in 0..n {
                let b = ((i*17+2304) % 13) > 8;
                let v = ((i*7+13) % (1<<width)) as u8;
                mask.set_bit(i, b);
                bs.set_value(i, v);

                if b { sum1 += v as u64; }
            }

            let sum2 = bs.sum_masked(&mask.into_bitset(n));

            println!("width {}: sums: {} / {}", width, sum1, sum2);
            assert_eq!(sum1, sum2);
        }
    }

    #[test]
    fn test_bitslice_sum_masked_and() {
        let n = 10_000;
        for &width in &[1u8, 2u8, 4u8] {
            let mut bs = BitSlice::new(n, width);
            let mut mask1 = BitVec::zero_bits(n);
            let mut mask2 = BitVec::zero_bits(n);

            assert_eq!(bs.sum(), 0);

            let mut sum1: u64 = 0;
            let mut sum2: u64 = 0;
            for i in 0..n {
                let b1 = ((i*17+2304) % 13) > 8;
                let b2 = ((i*23+2304) % 13) > 8;
                let v = ((i*7+13) % (1<<width)) as u8;
                mask1.set_bit(i, b1);
                mask2.set_bit(i, b2);
                bs.set_value(i, v);

                if b1 && b2 { sum1 += v as u64; }
                if b1 && !b2 { sum2 += v as u64; }
            }

            let m1 = mask1.into_bitset(n);
            let m2 = mask2.into_bitset(n);

            let sum3 = bs.sum_masked_and(&m1, &m2);
            let sum4 = bs.sum_masked_andnot(&m1, &m2);

            println!("width {}:   test sums: {:4} / {:4}", width, sum1, sum2);
            println!("width {}: actual sums: {:4} / {:4}", width, sum3, sum4);
            assert_eq!(sum1, sum3);
            assert_eq!(sum2, sum4);
        }
    }
}
