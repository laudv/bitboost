/*
 * Copyright (c) DTAI - KU Leuven - All rights reserved.
 * Proprietary, do not copy or distribute without permission.
 * Written by Laurens Devos, 2019
*/

use num::{Integer, One};

use std::mem::{size_of};
use std::slice;
use std::ops::{BitAnd, BitOr, Shr, Shl, Not, Deref, DerefMut};

use std::convert::From;






// - Utilities ------------------------------------------------------------------------------------

pub fn get_bit<T>(bits: T, pos: u8) -> bool
where T: Integer + One + Shr<Output=T> + BitAnd<Output=T> + From<u8> {
    T::one() & (bits >> T::from(pos)) == T::one()
}

pub fn enable_bit<T>(bits: T, pos: u8) -> T
where T: Integer + One + Shl<Output=T> + BitOr<Output=T> + From<u8> {
    bits | T::one() << T::from(pos)
}

pub fn disable_bit<T>(bits: T, pos: u8) -> T
where T: Integer + One + Not<Output=T> + Shl<Output=T> + BitAnd<Output=T> + From<u8> {
    bits & !(T::one() << T::from(pos))
}

pub fn set_bit<T>(bits: T, pos: u8, value: bool) -> T
where T: Integer + One + Not<Output=T> + Shl<Output=T> + BitAnd<Output=T> + BitOr<Output=T> + From<u8> {
    if value { enable_bit(bits, pos) }
    else     { disable_bit(bits, pos) }
}

pub fn get_blockpos<T: Integer>(bit_index: usize) -> usize {
     bit_index / (size_of::<T>() * 8)
}

pub fn get_bitpos<T: Integer>(bit_index: usize) -> u8 {
    (bit_index % (size_of::<T>() * 8)) as u8
}





// - Aligned bit block type -----------------------------------------------------------------------

const BITBLOCK_BYTES: usize = 32; // 32*8 = 256 bits per block

/// Properly aligned bit blocks (cache boundary Intel 64B)
#[repr(C, align(32))]
#[derive(Clone)]
pub struct BitBlock {
    bytes: [u8; BITBLOCK_BYTES],
}

impl BitBlock {
    pub fn nbytes() -> usize { size_of::<BitBlock>() }
    pub fn nbits() -> usize { Self::nbytes() * 8 }

    pub fn zeros() -> BitBlock {
        BitBlock { bytes: [0u8; BITBLOCK_BYTES] }
    }

    pub fn ones() -> BitBlock {
        BitBlock { bytes: [0xFFu8; BITBLOCK_BYTES] }
    }

    pub fn is_zero(&self) -> bool {
        let n = Self::nbytes() / size_of::<u128>();
        let ptr = self.as_ptr() as *const u128;
        for i in 0..n {
            if unsafe { *ptr.add(i) != 0 } { return false; }
        }
        return true;
    }

    /// Construct a BitBlock from an iterator of integers. Returns a tuple containing whether the
    /// iterator was fully consumed, and the constructed BitBlock.
    pub fn from_iter<T, I>(iter: &mut I) -> (bool, BitBlock)
    where T: Copy + Integer,
          I: Iterator<Item = T>
    {
        let mut iter_done = false;
        let mut bb = BitBlock::zeros();
        {
            let bb_arr = bb.cast_mut::<T>();
            for i in 0..bb_arr.len() {
                if let Some(v) = iter.next() {
                    bb_arr[i] = v;
                } else {
                    iter_done = true;
                    break;
                }
            }
        }
        (iter_done, bb)
    }

    /// Construct a BitBlock from an iterator of bools. Returns a tuple containing whether the
    /// iterator was fully consumed, and the constructed BitBlock.
    pub fn from_bool_iter<I>(iter: &mut I) -> (bool, BitBlock)
    where I: Iterator<Item = bool> {
        let mut iter_done = false;
        let mut bb = BitBlock::zeros();
        for i in 0..BitBlock::nbits() {
            if let Some(b) = iter.next() {
                bb.set_bit(i, b);
            } else {
                iter_done = true;
                break;
            }
        }
        (iter_done, bb)
    }

    /// Constructs a BitBlock from right to left
    pub fn from_slice<'a, T>(v: &'a [T]) -> BitBlock
    where T: 'a + Copy + Integer {
        let mut iter = v.into_iter().map(|x| *x);
        Self::from_iter(&mut iter).1
    }

    pub fn from_4u64(a: u64, b: u64, c: u64, d: u64) -> BitBlock {
        let mut bb = BitBlock::zeros();
        {
            let bb_arr = bb.cast_mut::<u64>();
            bb_arr[0] = a;
            bb_arr[1] = b;
            bb_arr[2] = c;
            bb_arr[3] = d;
        }
        bb
    }

    pub fn get_bit(&self, index: usize) -> bool {
        let b = get_blockpos::<u8>(index);
        let i = get_bitpos::<u8>(index);
        get_bit(self.bytes[b], i)
    }

    pub fn enable_bit(&mut self, index: usize) {
        let b = get_blockpos::<u8>(index);
        let i = get_bitpos::<u8>(index);
        self.bytes[b] = enable_bit(self.bytes[b], i);
    }

    pub fn disable_bit(&mut self, index: usize) {
        let b = get_blockpos::<u8>(index);
        let i = get_bitpos::<u8>(index);
        self.bytes[b] = disable_bit(self.bytes[b], i);
    }

    pub fn set_bit(&mut self, index: usize, value: bool) {
        if value { self.enable_bit(index);  }
        else     { self.disable_bit(index); }
    }

    pub fn count_ones(&self) -> u32 {
        let mut count = 0u32;
        let ptr = self.as_ptr() as *const u64;
        for i in 0..(Self::nbytes() / size_of::<u64>()) {
            count += unsafe { (*ptr.add(i)).count_ones() };
        }
        count
    }

    pub fn blocks_required_for(nbits: usize) -> usize {
        let w = Self::nbits();
        let correction = if (nbits % w) == 0 { 0 } else { 1 };
        nbits / w + correction
    }

    pub fn cast<T: Integer>(&self) -> &[T] {
        let sz = Self::nbytes() / size_of::<T>();
        let ptr = self.bytes.as_ptr() as *const T;
        unsafe {
            slice::from_raw_parts(ptr, sz)
        }
    }

    pub fn cast_mut<T: Integer>(&mut self) -> &mut [T] {
        let sz = Self::nbytes() / size_of::<T>();
        let ptr = self.bytes.as_mut_ptr() as *mut T;
        unsafe {
            slice::from_raw_parts_mut(ptr, sz)
        }
    }

    pub fn as_ptr(&self) -> *const u8 {
        self.bytes.as_ptr()
    }

    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.bytes.as_mut_ptr()
    }
}

impl <T: Integer> From<T> for BitBlock {
    fn from(v: T) -> BitBlock {
        let mut bb = BitBlock::zeros();
        bb.cast_mut::<T>()[0] = v;
        bb
    }
}

impl Default for BitBlock {
    fn default() -> Self { BitBlock::zeros() }
}







// - Array of BitBlocks ---------------------------------------------------------------------------

pub struct BitBlocks {
    blocks: Vec<BitBlock>,
}

impl BitBlocks {
    pub fn empty() -> BitBlocks {
        BitBlocks {
            blocks: Vec::new(),
        }
    }

    pub fn zero_blocks(nblocks: usize) -> BitBlocks {
        assert!(nblocks > 0);
        BitBlocks {
            blocks: vec![BitBlock::zeros(); nblocks],
        }
    }

    pub fn one_blocks(nblocks: usize) -> BitBlocks {
        assert!(nblocks > 0);
        BitBlocks {
            blocks: vec![BitBlock::ones(); nblocks],
        }
    }

    pub fn zero_bits(nbits: usize) -> BitBlocks {
        let nblocks = BitBlock::blocks_required_for(nbits);
        Self::zero_blocks(nblocks)
    }

    pub fn one_bits(nbits: usize) -> BitBlocks {
        let nblocks = BitBlock::blocks_required_for(nbits);
        let mut blocks = Self::one_blocks(nblocks);

        //if let Some(last) = blocks.get_slice_mut(range).last_mut() {
        if let Some(last) = blocks.deref_mut().last_mut() {
            let u64s = last.cast_mut::<u64>();  // zero out the last bits
            let mut zeros = nblocks * BitBlock::nbits() - nbits;
            let mut i = u64s.len()-1;
            loop {
                if zeros >= 64 { u64s[i] = 0; }
                else           { u64s[i] >>= zeros; }

                if zeros > 64 { zeros -= 64; i -= 1; }
                else          { break; }
            }
        }

        blocks
    }

    pub fn from_iter<T, I>(nvalues: usize, mut iter: I) -> BitBlocks
    where T: Integer + Copy,
          I: Iterator<Item = T>,
    {
        let nblocks = BitBlock::blocks_required_for(nvalues * size_of::<T>() * 8);
        let mut blocks = Self::zero_blocks(nblocks);
        for i in 0..nblocks {
            let (_, block) = BitBlock::from_iter(&mut iter);
            blocks[i] = block;
        }
        blocks
    }

    pub fn from_bool_iter<I>(nbits: usize, mut iter: I) -> BitBlocks
    where I: Iterator<Item = bool>,
    {
        let nblocks = BitBlock::blocks_required_for(nbits);
        let mut blocks = Self::zero_blocks(nblocks);
        for i in 0..nblocks {
            let (_, block) = BitBlock::from_bool_iter(&mut iter);
            blocks[i] = block;
        }
        blocks
    }

    pub fn block_len<T: Integer>(&self) -> usize {
        let nblocks = self.blocks.len();
        nblocks * (BitBlock::nbytes() / size_of::<T>())
    }

    pub fn cast<T: Integer>(&self) -> &[T] {
        let sz = self.block_len::<T>();
        let ptr = self.as_ptr() as *const T;
        unsafe { slice::from_raw_parts(ptr, sz) }
    }

    pub fn cast_mut<T: Integer>(&mut self) -> &mut [T] {
        let ptr = self.as_mut_ptr() as *mut T;
        let sz = self.block_len::<T>();
        unsafe { slice::from_raw_parts_mut(ptr, sz) }
    }

    pub fn get<T: Integer>(&self, index: usize) -> &T {
        &self.cast::<T>()[index]
    }

    pub unsafe fn get_unchecked<T: Integer>(&self, index: usize) -> &T {
        safety_check!(index < self.block_len::<T>());
        let ptr = self.as_ptr() as *const T;
        &*ptr.add(index)
    }

    pub fn set<T: Integer + Copy>(&mut self, index: usize, value: T) {
        self.cast_mut::<T>()[index] = value;
    }

    pub unsafe fn set_unchecked<T: Integer + Copy>(&mut self, index: usize, value: T) {
        safety_check!(index < self.block_len::<T>());
        let ptr = self.as_mut_ptr() as *mut T;
        *&mut *ptr.add(index) = value;
    }

    pub fn resize(&mut self, nblocks: usize) {
        assert!(nblocks > 0);
        self.blocks.resize(nblocks, BitBlock::zeros());
    }

    pub fn reset(&mut self) {
        self.blocks.clear();
    }
}

impl Deref for BitBlocks {
    type Target = [BitBlock];
    fn deref(&self) -> &[BitBlock] { &self.blocks }
}

impl DerefMut for BitBlocks {
    fn deref_mut(&mut self) -> &mut [BitBlock] { &mut self.blocks }
}















// ------------------------------------------------------------------------------------------------


#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_bitblock0() {
        let zeros = BitBlock::zeros();
        for i in 0..BITBLOCK_BYTES {
            assert_eq!(0, zeros.cast::<u8>()[i]);
        }

        let ones = BitBlock::ones();
        for i in 0..BITBLOCK_BYTES {
            assert_eq!(0xFF, ones.cast::<u8>()[i]);
        }
    }

    #[test]
    fn test_bitblock1() {
        let block1 = BitBlock::from_4u64(0b1001, 0, 0, 0);
        assert_eq!(BitBlock::nbits(), 256);
        assert_eq!(block1.get_bit(0), true);
        assert_eq!(block1.get_bit(1), false);
        assert_eq!(block1.get_bit(2), false);
        assert_eq!(block1.get_bit(3), true);
        assert_eq!(block1.get_bit(4), false);
        assert_eq!(block1.get_bit(63), false);

        let mut block2 = block1.clone();
        block2.set_bit(63, true);
        assert_eq!(block2.cast::<u64>()[0], 0x8000000000000009);
        assert_eq!(block2.get_bit(63), true);
    }

    #[test]
    fn test_bitblock2() {
        let mut block = BitBlock::from_4u64(0x0807060504030201, 0x0807060504030201,
                                            0x0807060504030201, 0x0807060504030201);
        for (i, u) in block.cast::<u8>().into_iter().enumerate() {
            assert_eq!((i as u8) % 8 + 1, *u);
        }

        block.enable_bit(129); // byte 16, was 0b01, now 0b11
        assert_eq!(block.cast::<u8>()[16], 0b11);
        block.enable_bit(130); // byte 16, was 0b11, now 0b111
        assert_eq!(block.cast::<u8>()[16], 0b111);
    }

    #[test]
    fn test_bitblock3() {
        let bits = [true, true, false, true, false, false, true];
        let mut iter = bits.iter().map(|b| *b);
        let (done, bb) = BitBlock::from_bool_iter(&mut iter);
        assert!(done);
        assert_eq!(bb.cast::<u8>()[0], 0b1001011); // reverse order, bit 0 first!
    }

    #[test]
    fn test_bitblock4() {
        let mut bb = BitBlock::zeros();
        bb.set_bit(10, true);
        assert!(bb.get_bit(10) == true);
        assert!(bb.cast::<u64>()[0] == 0x400);
        bb.set_bit(0, true);
        assert!(bb.get_bit(0) == true);
        bb.set_bit(1, true);
        assert!(bb.get_bit(1) == true);
        bb.set_bit(63, true);
        assert!(bb.get_bit(63) == true);
        bb.set_bit(64, true);
        assert!(bb.get_bit(64) == true);
        assert!(bb.cast::<u64>()[0] == 0x8000000000000403);
        assert!(bb.cast::<u64>()[1] == 0x1);
        bb.set_bit(63, false);
        assert!(bb.get_bit(63) == false);
        bb.set_bit(64, false);
        assert!(bb.get_bit(64) == false);
        assert!(bb.get_bit(1) == true);
        assert!(bb.cast::<u64>()[0] == 0x403);
        assert!(bb.cast::<u64>()[1] == 0x0);
        bb.set_bit(127, true);
        assert!(bb.get_bit(127) == true);
    }


    // bitblocks

    #[test]
    fn bitvec_basic() {
        let n = 10_000;

        let mut blocks = BitBlocks::from_iter::<u32, _>(n, 0u32..n as u32);
        for i in 0..n {
            assert_eq!(blocks.cast::<u32>()[i], i as u32);
            assert_eq!(*blocks.get::<u32>(i), i as u32);

            blocks.set::<u32>(i, 0);
            assert_eq!(blocks.cast::<u32>()[i], 0);
            assert_eq!(*blocks.get::<u32>(i), 0);

            blocks.cast_mut::<u32>()[i] = (n - i) as u32;
        }

        for i in 0..n {
            assert_eq!(blocks.cast::<u32>()[i], (n - i) as u32);
            assert_eq!(*blocks.get::<u32>(i), (n - i) as u32);
        }
    }

    #[test]
    fn bitvec_from_bool_iter() {
        let n = 10_000;
        let f = |k| k<n && k%13==1;
        let iter = (0..n).map(f);

        let blocks = BitBlocks::from_bool_iter(n, iter.clone());

        for (i, block) in blocks.iter().enumerate() {
            for j in 0..BitBlock::nbits() {
                let k = i*BitBlock::nbits() + j;
                let b = f(k);
                assert_eq!(b, block.get_bit(j));
            }
        }
    }

    #[test]
    fn bitvec_from_iter() {
        let n = 4367;
        let f = |i| if i >= n as u32 { 0 } else { 101*i+13 };

        let mut blocks = BitBlocks::from_iter(n, (0u32..n as u32).map(f));

        for (i, &b_u32) in blocks.cast::<u32>().iter().enumerate() {
            assert_eq!(b_u32, f(i as u32));
        }

        for i in 0..n {
            assert_eq!(*blocks.get::<u32>(i), f(i as u32));
        }

        for i in 0..n {
            unsafe {
                assert_eq!(*blocks.get_unchecked::<u32>(i), f(i as u32));
            }
        }

        for i in 0..n { blocks.set::<u32>(i, f(i as u32) + 10); }
        for i in 0..n { assert_eq!(*blocks.get::<u32>(i), f(i as u32) + 10); }
    }

    #[test]
    fn bitvec_cast_len() {
        let n = 13456;
        let f = |k| k<n && k%31==1;
        let iter = (0..n).map(f);

        let blocks = BitBlocks::from_bool_iter(n, iter);

        assert_eq!(blocks.len(), n / 256 + 1);
        assert_eq!(blocks.cast::<u128>().len(), blocks.len() * 2);
        assert_eq!(blocks.cast::<u64>().len(), blocks.len() * 4);
        assert_eq!(blocks.cast::<u32>().len(), blocks.len() * 8);
        assert_eq!(blocks.cast::<u16>().len(), blocks.len() * 16);
        assert_eq!(blocks.cast::<u8>().len(), blocks.len() * 32);

        for (i, qword) in blocks.cast::<u64>().iter().enumerate() {
            for j in 0..64 {
                let b = f(i*64 + j);
                assert_eq!(b, qword >> j & 0x1 == 0x1);
            }
        }
    }

    #[test]
    fn bitvec_zeros_end() {
        // allocate some memory
        let blocks = BitBlocks::from_iter(3, 10u32..13u32);
        assert_eq!(blocks.cast::<u32>()[1], 11);
        assert_eq!(blocks.cast::<u32>().iter().cloned().last().unwrap(), 0);

        for _ in 0..100 {
            let blocks = BitBlocks::from_iter(3, 10u32..13u32);
            for (i, &b_u32) in blocks.cast::<u32>().iter().enumerate() {
                if i < 3 { assert_eq!(b_u32, (10+i) as u32); }
                else     { assert_eq!(b_u32, 0); }
            }

            let blocks = BitBlocks::from_bool_iter(32, (0..32).map(|_| true));
            for (i, &b_u32) in blocks.cast::<u32>().iter().enumerate() {
                if i == 0 { assert_eq!(b_u32, 0xFFFFFFFF); }
                else      { assert_eq!(b_u32, 0); }
            }
        }
    }

    #[test]
    fn bitvec_one_bits() {
        let blocks = BitBlocks::one_bits(50);

        let v = blocks.cast::<u32>();
        assert_eq!(v.len(), 8);
        assert_eq!(v[0], 0xFFFFFFFF);
        assert_eq!(v[1], 0x3FFFF);
        assert_eq!(v[2], 0);
        assert_eq!(v[3], 0);
        assert_eq!(v[4], 0);
        assert_eq!(v[5], 0);
        assert_eq!(v[6], 0);
        assert_eq!(v[7], 0);
    }
}
