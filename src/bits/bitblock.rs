use std::mem::size_of;

pub trait BitBlockOps: Sized {
    fn nbits() -> usize { size_of::<Self>() * 8 }
    fn get_bit(self, index: usize) -> bool;
    fn enable_bit(self, index: usize) -> Self;
    fn disable_bit(self, index: usize) -> Self;
    fn set_bit(self, index: usize, value: bool) -> Self {
        if value { self.enable_bit(index) }
        else     { self.disable_bit(index) }
    }

    /// The minimum number of blocks required to store `nbits` bits.
    fn nblocks(nbits: usize) -> usize {
        let w = Self::nbits();
        let correction = if nbits % w == 0 { 0 } else { 1 };
        nbits / w + correction
    }
}

pub type BitBlock = u64;

impl BitBlockOps for BitBlock {
    fn get_bit(self, index: usize) -> bool     { 0x1 & (self >> index) == 0x1 }
    fn enable_bit(self, index: usize) -> Self  { self | (0x1 << index) }
    fn disable_bit(self, index: usize) -> Self { self & !(0x1 << index) }
}

#[cfg(test)]
mod test {
    use super::{BitBlockOps, BitBlock};

    #[test]
    fn test_bitblock_u64() {
        let block1 = 0b1001;
        assert_eq!(BitBlock::nbits(), 64);
        assert_eq!(block1.get_bit(0), true);
        assert_eq!(block1.get_bit(1), false);
        assert_eq!(block1.get_bit(2), false);
        assert_eq!(block1.get_bit(3), true);
        assert_eq!(block1.get_bit(4), false);
        assert_eq!(block1.get_bit(63), false);

        let block2 = block1.set_bit(63, true);
        assert_eq!(block2, 0x8000000000000009);
        assert_eq!(block2.get_bit(63), true);
    }
}
