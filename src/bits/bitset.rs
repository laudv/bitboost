//use std::fmt::{Debug, Result as FmtResult, Formatter};
use std::ops::{Deref, DerefMut};

use rand::prelude::*;
use rand::distributions::Bernoulli;

use bits::BitVec;

pub struct BitSet {
    vec: BitVec,
    //one_count: u64,
}

impl BitSet {

    pub fn falses(nbits: usize) -> BitSet {
        let vec = BitVec::zero_bits(nbits);
        Self::from_bitvec(vec)
    }

    pub fn trues(nbits: usize) -> BitSet {
        let vec = BitVec::one_bits(nbits);
        Self::from_bitvec(vec)
    }

    pub fn from_bool_iter<I>(iter: I) -> BitSet
    where I: Iterator<Item = bool> {
        let vec = BitVec::from_bool_iter(iter);
        Self::from_bitvec(vec)
    }

    pub fn from_bitvec(vec: BitVec) -> BitSet {
        BitSet {
            vec: vec,
        }
    }

    pub fn random(nbits: usize, frac1: f64) -> BitSet {
        let mut rng = thread_rng();
        let dist = Bernoulli::new(frac1);
        BitSet::from_bool_iter(rng.sample_iter(&dist).take(nbits))
    }

    pub fn count_ones_popcnt(&self) -> u64 {
        let mut count = 0;
        for i in 0..self.vec.nblocks() {
            count += self.vec.get_block(i).count_ones() as u64;
        }
        count
    }

    pub fn count_ones_avx2(&self) -> u64 {
        super::sum_simd::bitset_count_ones(&self.vec)
    }

    pub fn count_ones(&self) -> u64 {
        self.count_ones_avx2()
    }
}

impl Deref for BitSet {
    type Target = BitVec;
    fn deref(&self) -> &BitVec { &self.vec }
}

impl DerefMut for BitSet {
    fn deref_mut(&mut self) -> &mut BitVec { &mut self.vec }
}


//impl Debug for BitSet {
//    fn fmt(&self, f: &mut Formatter) -> FmtResult {
//        writeln!(f, "BitSet[nblocks={}]", self.nblocks())?;
//        for i in 0..self.nblocks() {
//            writeln!(f, "{:7}: {:064b}", i, self.blocks[i])?;
//        }
//        Ok(())
//    }
//}








#[cfg(test)]
mod test {
    use bits::BitSet;

    #[test]
    fn test_bitset() {
        let bs = BitSet::from_bool_iter((0..1024).map(|i| i%7==0));
        assert_eq!(147, bs.count_ones());
    }

    #[test]
    fn test_bitset_count_ones() {
        let n = 10_000;
        let frac1 = 0.25;

        for _ in 0..10 {
            let bs = BitSet::random(n, frac1);
            let a = bs.count_ones_popcnt();
            let b = bs.count_ones_avx2();
            println!("{} - {} = {}", a, b, a as i64 - b as i64);
            assert_eq!(a, b);
        }
    }
}
