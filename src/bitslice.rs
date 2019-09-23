/*
 * Copyright 2019 DTAI Research Group - KU Leuven.
 * License: Apache License 2.0
 * Author: Laurens Devos
*/

use std::borrow::{Borrow, BorrowMut};
use std::marker::PhantomData;
use std::mem::size_of;

use crate::NumT;
use crate::bitblock::{BitBlock, BitBlocks, get_bit, set_bit, get_bitpos, get_blockpos};



/// Bitslice memory layout.
pub trait BitsliceLayout {
    /// Number of bit lanes in Bitslice.
    fn width() -> usize;

    /// Number of consecutive 32-bit blocks of same order/significance.
    fn superblock_width() -> usize { BitBlock::nbytes() / (size_of::<u32>() * Self::width()) }

    /// The number of unique values that can be represented by this BitSlice.
    fn nunique_values() -> usize { 1 << Self::width() }

    fn linproj(value: NumT, count: NumT, (lo, hi): (NumT, NumT)) -> NumT {
        let maxval = (Self::nunique_values() - 1) as NumT;
        (value / maxval) * (hi - lo) + count * lo
    }
}

macro_rules! bitslice_layout {
    ($name:ident, $width:expr) => {
        pub struct $name;
        impl BitsliceLayout for $name {
            #[inline(always)]
            fn width() -> usize { $width }
        }
    }
}

bitslice_layout!(BitsliceLayout1, 1);
bitslice_layout!(BitsliceLayout2, 2);
bitslice_layout!(BitsliceLayout4, 4);
bitslice_layout!(BitsliceLayout8, 8);






pub struct Bitslice {
    blocks: BitBlocks,
}

pub struct BitsliceWithLayout<B, L>
where B: Borrow<BitBlocks>,
      L: BitsliceLayout,
{
    blocks: B,
    _layout: PhantomData<L>,
}

impl Bitslice {
    pub fn new() -> Bitslice {
        Bitslice {
            blocks: BitBlocks::empty(),
        }
    }

    pub fn nblocks(&self) -> usize {
        self.blocks.len()
    }

    pub fn with_layout<L>(&self) -> BitsliceWithLayout<&BitBlocks, L>
    where L: BitsliceLayout {
        BitsliceWithLayout::for_bitblocks(&self.blocks)
    }

    pub fn with_layout_mut<L>(&mut self) -> BitsliceWithLayout<&mut BitBlocks, L>
    where L: BitsliceLayout {
        BitsliceWithLayout::for_bitblocks(&mut self.blocks)
    }

    pub fn as_bitblocks(&self) -> &BitBlocks {
        &self.blocks
    }
}

impl <B, L> BitsliceWithLayout<B, L>
where B: Borrow<BitBlocks>,
      L: BitsliceLayout,
{
    pub fn for_bitblocks(bitblocks: B) -> Self {
        BitsliceWithLayout {
            blocks: bitblocks,
            _layout: PhantomData,
        }
    }

    pub fn resize(&mut self, nvalues: usize)
    where B: BorrowMut<BitBlocks> {
        let nblocks = BitBlock::blocks_required_for(nvalues) * L::width();
        self.blocks.borrow_mut().resize(nblocks);
    }

    pub fn nblocks(&self) -> usize {
        self.blocks.borrow().len()
    }

    ///            --- 256 bit --- | --- 256 bit --- | ...
    /// width=1: [ 1 1 1 1 1 1 1 1 | 1 1 1 1 1 1 1 1 | ... ]
    /// width=2: [ 1 1 1 1 2 2 2 2 | 1 1 1 1 2 2 2 2 | ... ]
    /// width=4: [ 1 1 2 2 4 4 8 8 | 1 1 2 2 4 4 8 8 | ... ]
    /// width=8: [ 1 2 3 4 5 6 7 8 | 1 2 3 4 5 6 7 8 | ... ]
    /// width=1:   - - - - - - - -  superblock_sz = 8
    /// width=2:   - - - -          superblock_sz = 4
    /// width=4:   - -              superblock_sz = 2
    /// width=4:   -                superblock_sz = 1
    fn get_indices(blockpos_u32: usize) -> (usize, usize) {
        let superblock_i = blockpos_u32 / L::superblock_width(); // index of the superblock
        let superblock_j = blockpos_u32 % L::superblock_width(); // index in the superblock
        (superblock_i, superblock_j)
    }

    fn get_bit_indices(bit_index: usize) -> (usize, u8) {
        let blockpos_u32 = get_blockpos::<u32>(bit_index); // global u32 index
        let bitpos_u32 = get_bitpos::<u32>(bit_index);     // bit index in u32
        (blockpos_u32, bitpos_u32)
    }

    //fn get_bit_indices(index: usize) -> (usize, usize, u8) {
    //    let blockpos_u32 = get_blockpos::<u32>(index); // global u32 index
    //    let bitpos_u32 = get_bitpos::<u32>(index);     // bit index in u32
    //    let superblock = Self::get_indices(blockpos_u32); // global u32 index -> superblock indices
    //    (superblock.0, superblock.1, bitpos_u32)
    //}

    /// Compute the linear u32 index into the BitSlice vec.
    fn bitslice_blockpos_u32(superblock_i: usize, superblock_j: usize, lane: usize) -> usize {
         (superblock_i * 8) + (lane * L::superblock_width()) + superblock_j
    }

    /// Same as `get_block`, but w/o bounds checking
    pub unsafe fn get_block_unchecked(&self, blockpos_u32: usize, lane: usize) -> u32 {
        let (superblock_i, superblock_j) = Self::get_indices(blockpos_u32);
        let slice = self.blocks.borrow().cast::<u32>();
        *slice.get_unchecked(Self::bitslice_blockpos_u32(superblock_i, superblock_j, lane))
    }

    /// Same as `get_block_mut`, but w/o bounds checking
    pub unsafe fn get_block_unchecked_mut(&mut self, blockpos_u32: usize, lane: usize) -> &mut u32
    where B: BorrowMut<BitBlocks> {
        let (superblock_i, superblock_j) = Self::get_indices(blockpos_u32);
        let slice = self.blocks.borrow_mut().cast_mut::<u32>();
        slice.get_unchecked_mut(Self::bitslice_blockpos_u32(superblock_i, superblock_j, lane))
    }

    /// Get the bits representing the `lane`'th significant bit for the 32 values in block
    /// `blockpos_u32`.
    pub fn get_block(&self, blockpos_u32: usize, lane: usize) -> u32 {
        let (superblock_i, superblock_j) = Self::get_indices(blockpos_u32);
        let slice = self.blocks.borrow().cast::<u32>();
        slice[Self::bitslice_blockpos_u32(superblock_i, superblock_j, lane)]
    }

    /// Get a mutable references to the bits representing the `lane`'th significant bit for the 32
    /// values in block `blockpos_u32`.
    pub fn get_block_mut(&mut self, blockpos_u32: usize, lane: usize) -> &mut u32
    where B: BorrowMut<BitBlocks> {
        let (superblock_i, superblock_j) = Self::get_indices(blockpos_u32);
        let slice = self.blocks.borrow_mut().cast_mut::<u32>();
        &mut slice[Self::bitslice_blockpos_u32(superblock_i, superblock_j, lane)]
    }

    pub fn get_value(&self, index: usize) -> u8 {
        let mut res = 0;
        let (blockpos, bitpos) = Self::get_bit_indices(index);
        for lane in 0..L::width() {
            let bits = self.get_block(blockpos, lane);
            let b = get_bit(bits, bitpos);
            res = set_bit(res, lane as u8, b);
        }
        res
    }

    pub fn set_value(&mut self, index: usize, value: u8)
    where B: BorrowMut<BitBlocks> {
        let (blockpos, bitpos) = Self::get_bit_indices(index);
        for lane in 0..L::width() {
            let bits = self.get_block_mut(blockpos, lane);
            let b = get_bit(value, lane as u8);
            *bits = set_bit(*bits, bitpos, b);
        }
    }

    pub fn get_scaled_value(&self, index: usize, bounds: (NumT, NumT)) -> NumT {
        L::linproj(self.get_value(index) as NumT, 1.0, bounds)
    }

    pub fn set_scaled_value(&mut self, index: usize, value: NumT, (lo, hi): (NumT, NumT)) -> u8
    where B: BorrowMut<BitBlocks> {
        let maxval = (L::nunique_values() - 1) as NumT;
        let v0 = NumT::min(hi, NumT::max(lo, value));
        let v1 = ((v0 - lo) / (hi - lo)) * maxval;
        let v2 = v1.round() as u8;
        self.set_value(index, v2);
        v2 // return the discretized value
    }
}










// ------------------------------------------------------------------------------------------------

#[cfg(test)]
mod test {

    const BOUNDS: (NumT, NumT) = (0.0, 1.0);

    use super::*;

    #[test]
    fn bitslice_test_get_set_value() {
        bitslice_get_set_value::<BitsliceLayout1>();
        bitslice_get_set_value::<BitsliceLayout2>();
        bitslice_get_set_value::<BitsliceLayout4>();
        bitslice_get_set_value::<BitsliceLayout8>();
    }

    fn bitslice_get_set_value<L>()
    where L: BitsliceLayout {
        let n = 20_000;
        let scale = (L::nunique_values() - 1) as NumT;

        let mut bitslice = Bitslice::new();
        let mut view = bitslice.with_layout_mut::<L>();
        view.resize(n);

        for i in 0..n {
            let k = ((101*i+37) % L::nunique_values() as usize) as u8;
            view.set_value(i, k);
            assert_eq!(k, view.get_value(i));

            view.set_value(i, 0);
            assert_eq!(0, view.get_value(i));

            let u = k as NumT / scale;
            view.set_scaled_value(i, u, BOUNDS);
            assert_eq!(u, view.get_scaled_value(i, BOUNDS));
            assert_eq!(k, view.get_value(i));
        }
    }
}
