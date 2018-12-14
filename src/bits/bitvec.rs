use std::mem::size_of;
use std::ops::{Deref, DerefMut};
use std::slice;

use num::Integer;

use bits::BitBlock;

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
        Self::one_blocks(nblocks)
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
    use bits::BitBlock;
    use bits::BitVec;

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
}
