use std::marker::PhantomData;

use bits::{BitBlock, BitVec};
use bits::bitblock::{get_bit, set_bit, get_blockpos, get_bitpos};

pub trait BitSliceLayout {
    fn width() -> usize;
    fn weights(i: usize) -> u64;
}

//macro_rules! impl_bitslice_width {
//    ($name:ident, $width:expr, $weights:expr) => {
//        pub struct $name;
//        impl BitSliceLayout for $name {
//            fn width() -> usize { $width }
//            fn weights(i: usize) -> u64 { $weights[i] }
//        }
//    }
//}
//
//impl_bitslice_width!(BitSlice1_1, 1, [1]);
//impl_bitslice_width!(BitSlice2_2, 2, [1, 2]);
//impl_bitslice_width!(BitSlice4_2, 4, [1, 2, 4, 8]);
//impl_bitslice_width!(BitSlice8_1, 8, [1, 2, 4, 8, 16, 32, 64, 128]);




// ------------------------------------------------------------------------------------------------

pub struct BitSlice<W: BitSliceLayout> {
    vec: BitVec,
    _marker: PhantomData<W>,
}

impl <W> BitSlice<W>
where W: BitSliceLayout {
    pub fn new(nbits: usize) -> BitSlice<W> {
        let nblocks = BitBlock::blocks_required_for(nbits);
        let vec = BitVec::zero_blocks(nblocks * W::width());

        BitSlice {
            vec: vec,
            _marker: PhantomData,
        }
    }

    pub fn random(nbits: usize) -> BitSlice<W> {
        use rand::prelude::*;
        use rand::distributions::Uniform;

        let mask = (1 << W::width()) - 1;
        let mut rng = thread_rng();
        let dist = Uniform::new(0u8, 0x23);
        let mut slice = BitSlice::new(nbits);

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
        for k in 0..W::width() {
            let bits = vec_u32[i * W::width() + k];
            let b = get_bit(bits, j);
            res = set_bit(res, k as u8, b);
        }
        res
    }

    pub fn set_value(&mut self, index: usize, value: u8) {
        let i = get_blockpos::<u64>(index);
        let j = get_bitpos::<u64>(index);
        let vec_u32 = self.vec.cast_mut::<u64>();
        for k in 0..W::width() {
            let bits = &mut vec_u32[i * W::width() + k];
            let b = get_bit(value, k as u8);
            *bits = set_bit(*bits, j, b);
        }
    }

    pub fn nblocks(&self) -> usize { self.vec.nblocks() }
    pub fn nbytes(&self) -> usize { self.vec.nbytes() }
    pub fn nbits(&self) -> usize { self.vec.nbits() }
    pub fn width(&self) -> u8 { W::width() as u8 }
    pub fn nunique_values(&self) -> u8 { 1 << W::width() }

    pub fn sum_masked_and(&self, mask1: &BitVec, mask2: &BitVec) -> u64 {
        unimplemented!()
    }

    pub fn sum_masked_andnot(&self, mask1: &BitVec, mask2: &BitVec) -> u64 {
        unimplemented!();
    }

    /// Compute (left count, right count, sum_left, sum_right)
    pub fn sum_filtered(&self, parent: &BitVec, left: &BitVec) -> (u64, u64, u64, u64) {
        debug_assert!(self.width() == 4); // only implemented for 4 bits for now

        let mut left_count = 0u64;
        let mut right_count = 0u64;
        let mut left_sum_counts = [0u64; 4];
        let mut right_sum_counts = [0u64; 4];

        let n = self.nblocks();
        let blocks = self.vec.cast::<u64>();
        let parent_blocks = parent.cast::<u64>();
        let left_blocks = left.cast::<u64>();

        assert_eq!(blocks.len(), 4 * n);
        assert_eq!(parent_blocks.len(),  n);
        assert_eq!(left_blocks.len(), n);

        for i in 0..n {
            let xp = unsafe { parent_blocks.get_unchecked(i) };
            let xl = unsafe { left_blocks.get_unchecked(i) };

            let mask_and = xp & xl;
            let mask_andnot = xp & !xl;

            left_count += mask_and.count_ones() as u64;
            right_count += mask_andnot.count_ones() as u64;

            for j in 0..4 {
                let b = unsafe { blocks.get_unchecked(4*i + j) };
                left_sum_counts[j] += (b & mask_and).count_ones() as u64;
                right_sum_counts[j] += (b & mask_andnot).count_ones() as u64;
            }
        }

        let left_sum =
                left_sum_counts[0] +
            2 * left_sum_counts[1] +
            4 * left_sum_counts[2] +
            8 * left_sum_counts[3];
        let right_sum =
                right_sum_counts[0] +
            2 * right_sum_counts[1] +
            4 * right_sum_counts[2] +
            8 * right_sum_counts[3];

        (left_count, right_count, left_sum, right_sum)
    }
}



//#[cfg(test)]
//mod test {
//    use bits::{BitSlice, BitVec};
//
//    #[test]
//    fn test_bitslice_len() {
//        let bs = BitSlice::new(100, 1u8);
//        assert_eq!(bs.nblocks(), 1);
//
//        let bs = BitSlice::new(100, 2u8);
//        assert_eq!(bs.nblocks(), 2);
//
//        let bs = BitSlice::new(257, 2u8);
//        assert_eq!(bs.nblocks(), 4);
//    }
//
//    #[test]
//    fn test_bitslice_set_value() {
//        let n = 10_000;
//        for &width in &[1u8, 2u8, 4u8] {
//            let mut bs = BitSlice::new(n, width);
//            for i in 0..n {
//                let v = ((i*7+13) % (1<<width)) as u8;
//                bs.set_value(i, v);
//                assert_eq!(bs.get_value(i), v);
//            }
//            for i in n..bs.nbits() / (width as usize) {
//                assert_eq!(bs.get_value(i), 0);
//            }
//        }
//    }
//
//    #[test] #[should_panic]
//    fn test_bitslice_wrong_width3() { BitSlice::new(10, 3); }
//
//    #[test] #[should_panic]
//    fn test_bitslice_wrong_width5() { BitSlice::new(10, 5); }
//
//    #[test]
//    fn test_bitslice_sum() {
//        let n = 10_000;
//        for &width in &[1u8, 2u8, 4u8] {
//            let mut bs = BitSlice::new(n, width);
//            assert_eq!(bs.sum(), 0);
//            let mut sum1: u64 = 0;
//            for i in 0..n {
//                let v = ((i*7+13) % (1<<width)) as u8;
//                bs.set_value(i, v);
//                sum1 += v as u64;
//            }
//            let sum2 = bs.sum();
//            println!("width {}: sums: {} / {}", width, sum1, sum2);
//            assert_eq!(sum1, sum2);
//        }
//
//    }
//
//    #[test]
//    fn test_bitslice_sum_masked() {
//        let n = 10_000;
//        for &width in &[1u8, 2u8, 4u8] {
//            let mut bs = BitSlice::new(n, width);
//            let mut mask = BitVec::zero_bits(n);
//
//            assert_eq!(bs.sum(), 0);
//
//            let mut sum1: u64 = 0;
//            for i in 0..n {
//                let b = ((i*17+2304) % 13) > 8;
//                let v = ((i*7+13) % (1<<width)) as u8;
//                mask.set_bit(i, b);
//                bs.set_value(i, v);
//
//                if b { sum1 += v as u64; }
//            }
//
//            let sum2 = bs.sum_masked(&mask.into_bitset(n));
//
//            println!("width {}: sums: {} / {}", width, sum1, sum2);
//            assert_eq!(sum1, sum2);
//        }
//    }
//
//    #[test]
//    fn test_bitslice_sum_masked_and() {
//        let n = 10_000;
//        for &width in &[1u8, 2u8, 4u8] {
//            let mut bs = BitSlice::new(n, width);
//            let mut mask1 = BitVec::zero_bits(n);
//            let mut mask2 = BitVec::zero_bits(n);
//
//            assert_eq!(bs.sum(), 0);
//
//            let mut sum1: u64 = 0;
//            let mut sum2: u64 = 0;
//            for i in 0..n {
//                let b1 = ((i*17+2304) % 13) > 8;
//                let b2 = ((i*23+2304) % 13) > 8;
//                let v = ((i*7+13) % (1<<width)) as u8;
//                mask1.set_bit(i, b1);
//                mask2.set_bit(i, b2);
//                bs.set_value(i, v);
//
//                if b1 && b2 { sum1 += v as u64; }
//                if b1 && !b2 { sum2 += v as u64; }
//            }
//
//            let m1 = mask1.into_bitset(n);
//            let m2 = mask2.into_bitset(n);
//
//            let sum3 = bs.sum_masked_and(&m1, &m2);
//            let sum4 = bs.sum_masked_andnot(&m1, &m2);
//
//            println!("width {}:   test sums: {:4} / {:4}", width, sum1, sum2);
//            println!("width {}: actual sums: {:4} / {:4}", width, sum3, sum4);
//            assert_eq!(sum1, sum3);
//            assert_eq!(sum2, sum4);
//        }
//    }
//
//    #[test]
//    fn test_bitslice_filtered() {
//        let n = 10_000;
//
//        let slice = BitSlice::random(n, 4);
//        let mask1 = BitVec::random(n, 0.25);
//        let mask2 = BitVec::random(n, 0.50);
//
//        let leftc1 = mask1.and(&mask2).count_ones();
//        let rightc1 = mask1.andnot(&mask2).count_ones();
//
//        let (leftc2, rightc2, lefts2, rights2) = slice.sum_filtered(&mask1, &mask2);
//
//        println!("{:?}", (leftc1, rightc1, 0, 0));
//        println!("{:?}", (leftc2, rightc2, lefts2, rights2));
//
//        assert_eq!(leftc1, leftc2);
//        assert_eq!(rightc1, rightc2);
//    }
//}
