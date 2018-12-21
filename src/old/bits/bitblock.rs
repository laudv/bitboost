use num::{Integer, One};

use std::mem::{size_of};
use std::slice;
use std::ops::{BitAnd, BitOr, Shr, Shl, Not};

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

/// Properly aligned bit blocks
#[repr(align(32))]
#[derive(Clone)]
pub struct BitBlock {
    bytes: [u8; BITBLOCK_BYTES],
}

impl BitBlock {
    pub fn nbytes() -> usize { size_of::<Self>() }
    pub fn nbits() -> usize { Self::nbytes() * 8 }

    pub fn zeros() -> BitBlock {
        let bytes: [u8; BITBLOCK_BYTES] = [0u8; BITBLOCK_BYTES];
        BitBlock { bytes: bytes }
    }

    pub fn ones() -> BitBlock {
        let bytes: [u8; BITBLOCK_BYTES] = [0xFFu8; BITBLOCK_BYTES];
        BitBlock { bytes: bytes }
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

#[cfg(test)]
mod test {
    use super::{BitBlock, BITBLOCK_BYTES};

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
}
