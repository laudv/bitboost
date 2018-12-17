use std::mem::size_of;
use std::ops::{Deref, DerefMut};
use std::slice;

use rand::prelude::*;
use rand::distributions::Bernoulli;

use num::Integer;

use bits::{BitBlock, BitSet};
use bits::simd;

pub struct BitVec {
    blocks: Vec<BitBlock>,
}

impl BitVec {

    pub fn zero_bits(nbits: usize) -> BitVec {
        assert!(nbits > 0);
        let nblocks = BitBlock::blocks_required_for(nbits);
        Self::zero_blocks(nblocks)
    }

    pub fn one_bits(nbits: usize) -> BitVec {
        assert!(nbits > 0);
        let nblocks = BitBlock::blocks_required_for(nbits);
        let mut vec = Self::one_blocks(nblocks);
        {
            let u64s = vec.cast_mut::<u64>();
            let mut zeros = nblocks * BitBlock::nbits() - nbits;
            let mut i = u64s.len()-1;
            loop {
                if zeros >= 64 { u64s[i] = 0; }
                else           { u64s[i] >>= zeros; }

                i -= 1;
                if zeros > 64 { zeros -= 64; }
                else          { break; }
            }
        }
        vec
    }

    pub fn zero_blocks(nblocks: usize) -> BitVec {
        assert!(nblocks > 0);
        let blocks = vec![BitBlock::zeros(); nblocks];
        BitVec { blocks: blocks }
    }

    pub fn one_blocks(nblocks: usize) -> BitVec {
        assert!(nblocks > 0);
        let blocks = vec![BitBlock::ones(); nblocks];
        BitVec { blocks: blocks }
    }

    pub fn from_bool_iter<I>(mut iter: I) -> BitVec
    where I: Iterator<Item = bool> {
        let mut blocks = Vec::new();
        loop {
            let (done, block) = BitBlock::from_bool_iter(&mut iter);
            blocks.push(block);
            if done { break; }
        }
        BitVec { blocks: blocks }
    }

    pub fn random(nbits: usize, frac1: f64) -> BitVec {
        let mut rng = thread_rng();
        let dist = Bernoulli::new(frac1);
        BitVec::from_bool_iter(rng.sample_iter(&dist).take(nbits))
    }

    pub fn nblocks(&self) -> usize { self.blocks.len() }
    pub fn nbytes(&self) -> usize { self.blocks.len() * BitBlock::nbytes() }
    pub fn nbits(&self) -> usize { self.blocks.len() * BitBlock::nbits() }

    pub fn get<T: Integer>(&self, index: usize) -> &T {
        if index >= self.nbytes() / size_of::<T>() {
            panic!("index out of bounds");
        }
        let ptr = self.as_ptr() as *const T;
        unsafe { &*ptr.add(index) }
    }

    pub fn get_mut<T: Integer>(&mut self, index: usize) -> &mut T {
        if index >= self.nbytes() / size_of::<T>() {
            panic!("index out of bounds");
        }
        let ptr = self.as_mut_ptr() as *mut T;
        unsafe { &mut *ptr.add(index) }
    }

    pub fn get_block(&self, index: usize) -> &BitBlock { &self.blocks[index] }
    pub fn get_block_mut(&mut self, index: usize) -> &mut BitBlock { &mut self.blocks[index] }

    pub fn get_bit(&self, index: usize) -> bool {
        let i = index / BitBlock::nbits();
        let j = index % BitBlock::nbits();
        self.get_block(i).get_bit(j)
    }

    pub fn set_bit(&mut self, index: usize, value: bool) {
        let i = index / BitBlock::nbits();
        let j = index % BitBlock::nbits();
        self.get_block_mut(i).set_bit(j, value);
    }

    pub fn iter<'a>(&'a self) -> slice::Iter<'a, BitBlock> {
        self.blocks.iter()
    }

    pub fn cast<T: Integer>(&self) -> &[T] {
        let sz = self.blocks.len() * (BitBlock::nbytes() / size_of::<T>());
        let ptr = self.blocks.as_ptr() as *const T;
        unsafe { 
            slice::from_raw_parts(ptr, sz)
        }
    }

    pub fn cast_mut<T: Integer>(&mut self) -> &mut [T] {
        let sz = self.blocks.len() * (BitBlock::nbytes() / size_of::<T>());
        let ptr = self.blocks.as_mut_ptr() as *mut T;
        unsafe { 
            slice::from_raw_parts_mut(ptr, sz)
        }
    }

    pub fn as_ptr(&self) -> *const u8 {
        debug_assert!(self.blocks.len() > 0);
        self.blocks.as_ptr() as *const u8
    }

    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        debug_assert!(self.blocks.len() > 0);
        self.blocks.as_mut_ptr() as *mut u8
    }

    pub fn count_ones_popcnt(&self) -> u64 {
        let mut count = 0;
        for i in 0..self.nblocks() {
            count += self.get_block(i).count_ones() as u64;
        }
        count
    }

    pub fn count_ones_avx2(&self) -> u64 {
        simd::bitvec_count_ones(&self)
    }

    pub fn count_ones(&self) -> u64 {
        self.count_ones_avx2()
    }

    pub fn count_and(&self, other: &BitVec) -> u64 {
        simd::bitvec_count_and(&self, other)
    }

    pub fn count_andnot(&self, other: &BitVec) -> u64 {
        simd::bitvec_count_andnot(&self, other)
    }

    /// Compute self & other
    pub fn and(&self, other: &BitVec) -> BitVec {
        let v = simd::and(&self, other);
        BitVec { blocks: v }
    }

    /// Compute self & ~other
    pub fn andnot(&self, other: &BitVec) -> BitVec {
        let v = simd::andnot(&self, other);
        BitVec { blocks: v }
    }

    pub fn into_bitset(self, used_nbits: usize) -> BitSet {
        BitSet::from_bitvec(used_nbits, self)
    }
}

impl Deref for BitVec {
    type Target = [BitBlock];
    fn deref(&self) -> &[BitBlock] { &self.blocks }
}

impl DerefMut for BitVec {
    fn deref_mut(&mut self) -> &mut [BitBlock] { &mut self.blocks }
}



#[cfg(test)]
mod test {
    use bits::{BitBlock, BitVec, BitSet};

    #[test]
    fn test_from_bool_iter() {
        let n = 13456;
        let f = |k| k<n && k%13==1;
        let iter = (0..n).map(f);

        let vec = BitVec::from_bool_iter(iter);

        for (i, block) in vec.iter().enumerate() {
            for j in 0..BitBlock::nbits() {
                let k = i*BitBlock::nbits() + j;
                let b = f(k);
                assert_eq!(b, block.get_bit(j));
            }
        }
    }

    #[test]
    fn test_cast_len() {
        let n = 13456;
        let f = |k| k<n && k%31==1;
        let iter = (0..n).map(f);

        let vec = BitVec::from_bool_iter(iter);

        assert_eq!(vec.nblocks(), n / 256 + 1);
        assert_eq!(vec.cast::<u128>().len(), vec.nblocks() * 2);
        assert_eq!(vec.cast::<u64>().len(), vec.nblocks() * 4);
        assert_eq!(vec.cast::<u32>().len(), vec.nblocks() * 8);
        assert_eq!(vec.cast::<u16>().len(), vec.nblocks() * 16);
        assert_eq!(vec.cast::<u8>().len(), vec.nblocks() * 32);

        for (i, qword) in vec.cast::<u64>().iter().enumerate() {
            for j in 0..64 {
                let b = f(i*64 + j);
                assert_eq!(b, qword >> j & 0x1 == 0x1);
            }
        }
    }

    #[test]
    fn test_get() {
        let n = 1000;
        let mut vec = BitVec::zero_bits(n*32);
        for i in 0..n {
            *vec.get_mut::<u32>(i) = i as u32;
        }
        for i in 0..n {
            assert_eq!(*vec.get::<u32>(i), i as u32);
        }
    }

    #[test]
    fn test_count_ones() {
        let n = 10_000;
        let frac1 = 0.25;

        for _ in 0..10 {
            let bs = BitSet::random(n, frac1).into_bitvec();
            let a = bs.count_ones_popcnt();
            let b = bs.count_ones_avx2();
            println!("{} - {} = {}", a, b, a as i64 - b as i64);
            assert_eq!(a, b);
        }
    }

    #[test]
    fn test_and_andnot1() {
        let n = 10_000;

        let f1 = |i: usize| ((i*17+2304) % 13) > 8;
        let f2 = |i: usize| ((i*23+2304) % 13) > 8;

        let vec1 = BitVec::from_bool_iter((0..n).map(f1));
        let vec2 = BitVec::from_bool_iter((0..n).map(f2));

        let mut c_and = 0;
        let mut c_andnot = 0;
        for i in 0..n {
            if f1(i) && f2(i) { c_and += 1; }
            if f1(i) && !f2(i) { c_andnot += 1; }
        }

        assert_eq!(vec1.and(&vec2).count_ones(), c_and);
        assert_eq!(vec1.andnot(&vec2).count_ones(), c_andnot);
    }

    #[test]
    fn test_and_andnot2() {
        let n = 10_000;
        let k = 10;

        for _ in 0..k {
            let vec1 = BitSet::random(n, 0.5);
            let vec2 = BitSet::random(n, 0.5);

            let mut c_and = 0;
            let mut c_andnot = 0;
            for i in 0..n {
                if vec1.get_bit(i) && vec2.get_bit(i) { c_and += 1 }
                if vec1.get_bit(i) && !vec2.get_bit(i) { c_andnot += 1 }
            }
            assert_eq!(vec1.and(&vec2).count_ones(), c_and);
            assert_eq!(vec1.andnot(&vec2).count_ones(), c_andnot);
        }
    }
}
