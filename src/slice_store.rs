use std::alloc;
use std::borrow::{Borrow, BorrowMut};
use std::fmt::Debug;
use std::mem::{size_of, align_of};
use std::ops::{Sub, Add, Deref, DerefMut};
use std::slice;
use std::marker::PhantomData;

use num::Integer;

use log::debug;

use crate::NumT;
use crate::bits::{BitBlock};
use crate::bits::bitblock::{get_bit, set_bit, get_bitpos, get_blockpos};
use crate::dataset::{Dataset, FeatureRepr};
use crate::simd;

pub type SliceRange = (u32, u32);
pub type BitVecRef<'a> = BitVec<&'a [BitBlock]>;
pub type BitVecMut<'a> = BitVec<&'a mut [BitBlock]>;
pub type BitSliceRef<'a, L> = BitSlice<&'a [BitBlock], L>;
pub type BitSliceMut<'a, L> = BitSlice<&'a mut [BitBlock], L>;

pub struct SliceStore<T>
where T: Clone {

    /// Storage for slices.
    buffer: Vec<T>,

    /// Freed slices: very simple design, but allows simple memory reuse w/o reallocation.
    free: Vec<SliceRange>,

    /// Align byte boundary.
    elem_align: usize,
}

impl <T> SliceStore<T>
where T: Clone {

    pub fn new(initial_cap: usize) -> Self {
        let align = usize::max(align_of::<T>(), size_of::<T>());
        Self::aligned(initial_cap, align)
    }

    pub fn aligned(initial_cap: usize, byte_align: usize) -> Self {
        assert!(byte_align % size_of::<T>() == 0, "invalid alignment");

        let elem_align = byte_align / size_of::<T>();
        SliceStore {
            buffer: Self::alloc(initial_cap, byte_align),
            free: Vec::new(),
            elem_align: elem_align,
        }
    }

    fn alloc(nelems: usize, byte_align: usize) -> Vec<T> {
        let nbytes = nelems * size_of::<T>();
        unsafe {
            let layout = alloc::Layout::from_size_align(nbytes, byte_align).unwrap();
            let ptr = alloc::alloc(layout) as *mut T;
            if ptr.is_null() { panic!("out of memory"); }

            assert!(ptr as usize % byte_align == 0);

            Vec::from_raw_parts(ptr, 0, nelems)
        }
    }

    pub fn get_slice(&self, r: SliceRange) -> &[T] {
        &self.buffer[r.0 as usize..r.1 as usize]
    }

    pub fn get_slice_mut(&mut self, r: SliceRange) -> &mut [T] {
        &mut self.buffer[r.0 as usize..r.1 as usize]
    }

    pub fn get_two_slices_mut(&mut self, r1: SliceRange, r2: SliceRange)
        -> (&mut [T], &mut [T])
    {
        debug_assert!(r1 != r2);
        if r1.0 < r2.0 {
            let (s1, s2) = self.buffer.split_at_mut(r2.0 as usize);
            (&mut s1[r1.0 as usize..r1.1 as usize], &mut s2[0..(r2.1-r2.0) as usize])
        } else {
            let (s1, s2) = self.buffer.split_at_mut(r1.0 as usize);
            (&mut s2[0..(r1.1-r1.0) as usize], &mut s1[r2.0 as usize..r2.1 as usize])
        }
    }

    pub fn alloc_slice(&mut self, len: u32, value: T) -> SliceRange {
        // Check if we can use a free range
        // Note that we forget the full length of a slice when a shorter slice replaces a longer
        // previously freed slice.
        for i in 0..self.free.len() {
            let r = self.free[i];
            if r.1 - r.0 >= len {
                debug!("Reusing slice! #free = {}", self.free.len());
                self.free.swap_remove(i);
                return (r.0, r.0 + len); // XXX information lost for next reuse!
            }
        }

        // allocate new memory in buffer
        let old_len_unaligned = self.buffer.len();
        let m = old_len_unaligned % self.elem_align;
        let old_len = if m == 0 { old_len_unaligned }
                      else { old_len_unaligned + self.elem_align - m };
        let new_len = old_len + len as usize;

        assert!(new_len < u32::max_value() as usize);
        debug_assert!(self.buffer.as_ptr() as usize % (self.elem_align * size_of::<T>()) == 0);

        self.buffer.resize(new_len, value);
        (old_len as u32, new_len as u32)
    }

    pub fn alloc_slice_default(&mut self, len: u32) -> SliceRange
    where T: Default {
        self.alloc_slice(len, T::default())
    }

    pub fn free_slice(&mut self, range: SliceRange) {
        self.free.push(range)
    }
}






// ------------------------------------------------------------------------------------------------

pub struct HistStore<T>
where T: Clone + Default {
    slice_store: SliceStore<T>,
    hist_layout: Vec<u32>,
}

impl <T> HistStore<T>
where T: Clone + Default {
    pub fn new<I>(bin_size_iter: I) -> Self
    where I: Iterator<Item = u32> {
        let mut hist_layout = Vec::new();
        let mut accum = 0;

        hist_layout.push(0);
        for bin_size in bin_size_iter {
            accum += bin_size;
            hist_layout.push(accum);
        }

        HistStore {
            slice_store: SliceStore::new(8192),
            hist_layout: hist_layout,
        }
    }

    pub fn for_dataset(dataset: &Dataset) -> Self {
        // Use the right amount of 'buckets' for each feature
        // Currently only categorical features; one bucket for each cat. feat. value.
        Self::new(dataset.features().iter().map(|f| {
            match f.get_repr() {
                Some(&FeatureRepr::CatFeature(card, _)) => card as u32,
                Some(&FeatureRepr::BitVecFeature(ref f)) => f.card as u32,
                _ => panic!("feat repr not supported by histogram"),
            }
        }))
    }

    fn get_histogram_range(&self, feat_id: usize) -> (usize, usize) {
        let lo = self.hist_layout[feat_id];
        let hi = self.hist_layout[feat_id + 1];
        (lo as usize, hi as usize)
    }

    pub fn get_hist(&self, hists_range: SliceRange, feat_id: usize) -> &[T] {
        let (lo, hi) = self.get_histogram_range(feat_id);
        &self.slice_store.get_slice(hists_range)[lo..hi]
    }

    pub fn get_hist_mut(&mut self, hists_range: SliceRange, feat_id: usize) -> &mut [T] {
        let (lo, hi) = self.get_histogram_range(feat_id);
        &mut self.slice_store.get_slice_mut(hists_range)[lo..hi]
    }
    
    pub fn hists_subtract(&mut self, parent_range: SliceRange, left_range: SliceRange,
                          right_range: SliceRange)
    where T: Sub<Output=T> {
        let (plo, phi) = parent_range;
        let (llo, _) = left_range;
        let (rlo, _) = right_range;

        let buffer = &mut self.slice_store.buffer;

        debug_assert_eq!(phi-plo, *self.hist_layout.last().unwrap());

        for i in 0..(phi-plo) {
            let parent = buffer[(plo+i) as usize].clone();
            let left   = buffer[(llo+i) as usize].clone();
            buffer[(rlo+i) as usize] = parent - left;
        }
    }

    pub fn sum_hist(&self, hists_range: SliceRange, feat_id: usize) -> T
    where T: Add<Output=T> + Default {
        let mut sum = T::default();
        let hist = self.get_hist(hists_range, feat_id);
        for v in hist {
            sum = sum + v.clone();
        }
        sum
    }

    pub fn debug_print(&self, hists_range: SliceRange)
    where T: Debug + Add<Output=T> + Default {
        println!("Histograms");
        for feat_id in 0..self.hist_layout.len()-1 {
            for (i, val) in self.get_hist(hists_range, feat_id).iter().enumerate() {
                println!("{:4}: {:?}", i, val);
            }
            println!("-- +: {:?}", self.sum_hist(hists_range, feat_id));
            println!();
        }
    }

    pub fn alloc_hists(&mut self) -> SliceRange {
        let total_bins = *self.hist_layout.last().unwrap();
        self.slice_store.alloc_slice(total_bins, T::default())
    }

    pub fn free_hists(&mut self, r: SliceRange) {
        self.slice_store.free_slice(r);
    }
}







// ------------------------------------------------------------------------------------------------

pub struct BitBlockStore {
    slice_store: SliceStore<BitBlock>,
}

impl BitBlockStore {
    pub fn new(initial_cap: usize) -> Self {
        let intel_cache_line = 64;
        let store = SliceStore::aligned(initial_cap, intel_cache_line);

        BitBlockStore {
            slice_store: store,
        }
    }

    pub fn alloc_zero_blocks(&mut self, nblocks: usize) -> SliceRange {
        debug_assert!(nblocks < u32::max_value() as usize);
        self.slice_store.alloc_slice(nblocks as u32, BitBlock::zeros())
    }

    pub fn alloc_zero_bits(&mut self, nbits: usize) -> SliceRange {
        let nblocks = BitBlock::blocks_required_for(nbits);
        self.alloc_zero_blocks(nblocks)
    }

    pub fn alloc_one_blocks(&mut self, nblocks: usize) -> SliceRange {
        debug_assert!(nblocks < u32::max_value() as usize);
        self.slice_store.alloc_slice(nblocks as u32, BitBlock::ones())
    }

    pub fn alloc_one_bits(&mut self, nbits: usize) -> SliceRange {
        let nblocks = BitBlock::blocks_required_for(nbits);
        let range = self.alloc_one_blocks(nblocks);

        if let Some(last) = self.slice_store.get_slice_mut(range).last_mut() {
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

        range
    }

    pub fn alloc_from_iter<T, I>(&mut self, nvalues: usize, mut iter: I) -> SliceRange
    where T: Integer + Copy,
          I: Iterator<Item = T>
    {
        let nblocks = BitBlock::blocks_required_for(nvalues * size_of::<T>() * 8);
        let range = self.alloc_zero_blocks(nblocks);
        let slice = self.slice_store.get_slice_mut(range);
        for i in 0..nblocks {
            let (_, block) = BitBlock::from_iter(&mut iter);
            slice[i] = block;
        }
        range
    }

    pub fn alloc_from_bits_iter<I>(&mut self, nbits: usize, mut iter: I) -> SliceRange
    where I: Iterator<Item = bool>
    {
        let nblocks = BitBlock::blocks_required_for(nbits);
        let range = self.alloc_zero_blocks(nblocks);
        let slice = self.slice_store.get_slice_mut(range);
        for i in 0..nblocks {
            let (_, block) = BitBlock::from_bool_iter(&mut iter);
            slice[i] = block;
        }
        range
    }

    pub fn alloc_zero_bitslice<I>(&mut self, nvalues: usize) -> SliceRange
    where I: BitSliceLayout {
        let nblocks = BitBlock::blocks_required_for(nvalues);
        self.alloc_zero_blocks(nblocks * I::width())
    }

    pub fn free_blocks(&mut self, range: SliceRange) {
        self.slice_store.free_slice(range);
    }

    pub fn get_bitvec(&self, range: SliceRange) -> BitVec<&[BitBlock]> {
        BitVec { blocks: self.slice_store.get_slice(range) }
    }

    pub fn get_bitvec_mut(&mut self, range: SliceRange) -> BitVec<&mut [BitBlock]> {
        BitVec { blocks: self.slice_store.get_slice_mut(range) }
    }

    pub fn get_two_bitvecs_mut(&mut self, r1: SliceRange, r2: SliceRange)
        -> (BitVecMut, BitVecMut)
    {
        let (s1, s2) = self.slice_store.get_two_slices_mut(r1, r2);
        let b1 = BitVec { blocks: s1 };
        let b2 = BitVec { blocks: s2 };
        (b1, b2)
    }

    pub fn get_bitslice<'a, L>(&'a self, range: SliceRange)
        -> BitSliceRef<L>
    where L: 'a + BitSliceLayout
    {
        debug_assert!((range.1 - range.0) as usize % L::width() == 0);
        BitSlice::new(self.get_bitvec(range))
    }

    pub fn get_bitslice_mut<'a, L>(&'a mut self, range: SliceRange)
        -> BitSliceMut<L>
    where L: 'a + BitSliceLayout
    {
        debug_assert!((range.1 - range.0) as usize % L::width() == 0);
        BitSlice::new(self.get_bitvec_mut(range))
    }

    pub fn get_two_bitslices_mut<'a, L>(&'a mut self, r1: SliceRange, r2: SliceRange)
        -> (BitSliceMut<'a, L>, BitSliceMut<'a, L>)
    where L: 'a + BitSliceLayout
    {
        let (v1, v2) = self.get_two_bitvecs_mut(r1, r2);
        (BitSlice::new(v1), BitSlice::new(v2))
    }
}


/// A slice of BitBlocks.
pub struct BitVec<B>
where B: Borrow<[BitBlock]> {
    blocks: B
}

impl <B> BitVec<B>
where B: Borrow<[BitBlock]> {
    pub fn block_len<T: Integer>(&self) -> usize {
        let blocks = self.blocks.borrow();
        blocks.len() * (BitBlock::nbytes() / size_of::<T>())
    }

    pub fn cast<T: Integer>(&self) -> &[T] {
        let sz = self.block_len::<T>();
        let ptr = self.as_ptr() as *const T;
        unsafe { slice::from_raw_parts(ptr, sz) }
    }

    pub fn cast_mut<T: Integer>(&mut self) -> &mut [T]
    where B: BorrowMut<[BitBlock]> {
        let ptr = self.as_mut_ptr() as *mut T;
        let sz = self.block_len::<T>();
        unsafe { slice::from_raw_parts_mut(ptr, sz) }
    }

    pub fn get_bit(&self, index: usize) -> bool {
        let i = index / BitBlock::nbits();
        let j = index % BitBlock::nbits();
        let blocks = self.blocks.borrow();
        blocks[i].get_bit(j)
    }

    pub fn set_bit(&mut self, index: usize, bit: bool)
    where B: BorrowMut<[BitBlock]> {
        let i = index / BitBlock::nbits();
        let j = index % BitBlock::nbits();
        let blocks = self.blocks.borrow_mut();
        blocks[i].set_bit(j, bit)
    }

    pub fn get<T: Integer>(&self, index: usize) -> &T {
        &self.cast::<T>()[index]
    }

    pub unsafe fn get_unchecked<T: Integer>(&self, index: usize) -> &T {
        let ptr = self.as_ptr() as *const T;
        &*ptr.add(index)
    }

    pub fn set<T: Integer + Copy>(&mut self, index: usize, value: T)
    where B: BorrowMut<[BitBlock]> {
        self.cast_mut::<T>()[index] = value;
    }

    pub unsafe fn set_unchecked<T: Integer + Copy>(&mut self, index: usize, value: T)
    where B: BorrowMut<[BitBlock]> {
        let ptr = self.as_mut_ptr() as *mut T;
        *&mut *ptr.add(index) = value;
    }
    
    pub fn count_ones_and(&self, other: &BitVec<B>) -> u64 {
        unsafe { simd::bitvec_count_and_uc(self, other) }
    }

    /// For each u32 mask in this bitset, look-up the corresponding u32 index in `indices` and use
    /// that index to find a second mask in `other`.
    pub unsafe fn count_ones_and_compr_unsafe(&self, indices: &BitVec<B>, other: &BitVec<B>)
        -> u64
    {
        simd::bitvec_count_and_c(self, indices, other)
    }

    pub fn count_ones_and_compr(&self, indices: &BitVec<B>, other: &BitVec<B>) -> u64 {
        let m = other.block_len::<u32>();
        assert!(indices.cast::<u32>().iter().all(|&i| (i as usize) < m));
        unsafe { self.count_ones_and_compr_unsafe(indices, other) }
    }
}

impl <B> Deref for BitVec<B>
where B: Borrow<[BitBlock]> {
    type Target = [BitBlock];
    fn deref(&self) -> &[BitBlock] { self.blocks.borrow() }
}

impl <B> DerefMut for BitVec<B>
where B: Borrow<[BitBlock]> + BorrowMut<[BitBlock]> {
    fn deref_mut(&mut self) -> &mut [BitBlock] { self.blocks.borrow_mut() }
}


/// BitSlice memory layout.
pub trait BitSliceLayout {
    /// Number of bit lanes in BitSlice.
    fn width() -> usize;

    /// Number of consecutive 32-bit blocks of same order.
    fn superblock_width() -> usize { (BitBlock::nbytes() / size_of::<u32>()) / Self::width() }

    /// The number of unique values that can be represented by this BitSlice.
    fn nunique_values() -> usize { 1 << Self::width() }

    fn linproj(value: NumT, count: NumT, (lo, hi): (NumT, NumT)) -> NumT {
        let maxval = (Self::nunique_values() - 1) as NumT;
        (value / maxval) * (hi - lo) + count * lo
    }
}

/// Low-bit value encodings for integer or real values.
pub struct BitSlice<B, L>
where B: Borrow<[BitBlock]>,
      L: BitSliceLayout {
    vec: BitVec<B>,
    _marker: PhantomData<L>,
}

impl <B, L> BitSlice<B, L>
where B: Borrow<[BitBlock]>,
      L: BitSliceLayout
{
    fn new(vec: BitVec<B>) -> BitSlice<B, L> {
        BitSlice {
            vec: vec,
            _marker: PhantomData,
        }
    }

    ///            --- 256 bit --- | --- 256 bit --- | ...
    /// width=1: [ 1 1 1 1 1 1 1 1 | 1 1 1 1 1 1 1 1 | ... ]
    /// width=2: [ 1 1 1 1 2 2 2 2 | 1 1 1 1 2 2 2 2 | ... ]
    /// width=4: [ 1 1 2 2 4 4 8 8 | 1 1 2 2 4 4 8 8 | ... ]
    /// width=1:   - - - - - - - -  superblock_sz = 8
    /// width=2:   - - - -          superblock_sz = 4
    /// width=4:   - -              superblock_sz = 2
    fn get_indices(blockpos_u32: usize) -> (usize, usize) {
        let superblock_i = blockpos_u32 / L::superblock_width(); // index of the superblock
        let superblock_j = blockpos_u32 % L::superblock_width(); // index in the superblock
        (superblock_i, superblock_j)
    }

    fn get_bit_indices(index: usize) -> (usize, usize, u8) {
        let blockpos_u32 = get_blockpos::<u32>(index); // global u32 index
        let bitpos_u32 = get_bitpos::<u32>(index);     // bit index in u32
        let superblock = Self::get_indices(blockpos_u32); // global u32 index -> superblock indices
        (superblock.0, superblock.1, bitpos_u32)
    }
    
    /// Compute the linear u32 index into the BitSlice vec.
    fn bitslice_blockpos_u32(superblock_i: usize, superblock_j: usize, lane: usize) -> usize {
         (superblock_i * 8) + (lane * L::superblock_width()) + superblock_j
    }

    pub fn get_value(&self, index: usize) -> u8 {
        let mut res = 0;
        let (sb, i, j) = Self::get_bit_indices(index);
        let vec_u32 = self.vec.cast::<u32>();
        for k in 0..L::width() {
            let bits = vec_u32[Self::bitslice_blockpos_u32(sb, i, k)];
            let b = get_bit(bits, j);
            res = set_bit(res, k as u8, b);
        }
        res

        //let mut res = 0;
        //let i = get_blockpos::<u32>(index);
        //let j = get_bitpos::<u32>(index);
        //let vec_u32 = self.vec.cast::<u32>();
        //for k in 0..L::width() {
        //    let bits = vec_u32[i * L::width() + k];
        //    let b = get_bit(bits, j);
        //    res = set_bit(res, k as u8, b);
        //}
        //res
    }

    pub fn set_value(&mut self, index: usize, value: u8)
    where B: BorrowMut<[BitBlock]> {
        let (sb, i, j) = Self::get_bit_indices(index);
        let vec_u32 = self.vec.cast_mut::<u32>();
        for k in 0..L::width() {
            let bits = &mut vec_u32[Self::bitslice_blockpos_u32(sb, i, k)];
            let b = get_bit(value, k as u8);
            *bits = set_bit(*bits, j, b);
        }

        //let i = get_blockpos::<u32>(index);
        //let j = get_bitpos::<u32>(index);
        //let vec_u32 = self.vec.cast_mut::<u32>();
        //for k in 0..L::width() {
        //    let bits = &mut vec_u32[i * L::width() + k];
        //    let b = get_bit(value, k as u8);
        //    *bits = set_bit(*bits, j, b);
        //}
    }

    pub fn get_scaled_value(&self, index: usize, bounds: (NumT, NumT)) -> NumT {
        L::linproj(self.get_value(index) as NumT, 1.0, bounds)
    }

    pub fn set_scaled_value(&mut self, index: usize, value: NumT, (lo, hi): (NumT, NumT))
    where B: BorrowMut<[BitBlock]> {
        let maxval = (L::nunique_values() - 1) as NumT;
        let v0 = NumT::min(hi, NumT::max(lo, value));
        let v1 = ((v0 - lo) / (hi - lo)) * maxval;
        let v2 = v1.round() as u8;
        //println!("{}, {}", x, v2);
        self.set_value(index, v2);
    }

    pub fn copy_block_from<I, BO>(&mut self, other: &BitSlice<BO, L>, from: usize, to: usize)
    where B: BorrowMut<[BitBlock]>,
          I: Integer,
          BO: Borrow<[BitBlock]>,
    {
        assert!(from < other.vec.block_len::<u32>() / L::width());
        assert!(to < self.vec.block_len::<u32>() / L::width());

        let (from_sb, from_i) = Self::get_indices(from);
        let (to_sb, to_i) = Self::get_indices(to);
        let from_base = other.vec.as_ptr() as *const u32;
        let to_base = self.vec.as_mut_ptr() as *mut u32;

        for k in 0..L::width() {
            let from_l = Self::bitslice_blockpos_u32(from_sb, from_i, k);
            let to_l = Self::bitslice_blockpos_u32(to_sb, to_i, k);

            unsafe { // should be safe because asserts
                let from_block = *from_base.add(from_l);
                *to_base.add(to_l) = from_block;
            }
        }
    }

    pub unsafe fn sum_masked_unsafe(&self, index: usize, mask: u32) -> u32 {
        debug_assert!(index < self.vec.block_len::<u32>() / L::width());

        let (sb, i) = Self::get_indices(index);
        let base = self.vec.as_ptr() as *const u32;

        let mut sum = 0;
        for k in 0..L::width() {
            let l = Self::bitslice_blockpos_u32(sb, i, k);
            let block = *base.add(l);
            let count = (block & mask).count_ones();
            let weight = 1 << k;
            sum += count * weight;
        }

        sum
    }

    pub fn sum_masked(&self, index: usize, mask: u32) -> u32 {
        assert!(index < self.vec.block_len::<u32>() / L::width());
        unsafe { self.sum_masked_unsafe(index, mask) }
    }

    pub unsafe fn sum_scaled_masked_unsafe(&self, index: usize, mask: u32, bounds: (NumT, NumT))
        -> (NumT, u32)
    {
        let count = mask.count_ones();
        let sum = self.sum_masked_unsafe(index, mask);
        (L::linproj(sum as NumT, count as NumT, bounds), count)
    }

    pub fn sum_scaled_masked(&self, index: usize, mask: u32, bounds: (NumT, NumT))
        -> (NumT, u32)
    {
        assert!(index < self.vec.block_len::<u32>() / L::width());
        unsafe { self.sum_scaled_masked_unsafe(index, mask, bounds) }
    }

    pub unsafe fn sum_all_masked2_unsafe(&self, nm: &BitVecRef, fm: &BitVecRef) -> u64 {
        let fs: [unsafe fn(&[BitBlock], &[BitBlock], &[BitBlock]) -> u64; 4] = [
            simd::btslce_summ1_nc,
            simd::btslce_summ2_nc,
            simd::btslce_summ1_nc, // invalid
            simd::btslce_summ4_nc];

        let f = fs[L::width() - 1];
        f(self, nm, fm)
    }

    pub fn sum_all_masked2(&self, nm: &BitVecRef, fm: &BitVecRef) -> u64 {
        assert_eq!(self.len(), nm.len() * L::width());
        assert_eq!(self.len(), fm.len() * L::width());
        unsafe { self.sum_all_masked2_unsafe(nm, fm) }
    }

    pub unsafe fn sum_all_masked2_compr_unsafe(&self, idxs: &BitVecRef, nm: &BitVecRef,
                                               fm: &BitVecRef) -> u64
    {
        let fs: [unsafe fn(&[BitBlock], &[BitBlock], &[BitBlock], &[BitBlock]) -> u64; 4] = [
            simd::btslce_summ1_c,
            simd::btslce_summ2_c,
            simd::btslce_summ1_c, // invalid
            simd::btslce_summ4_c];

        let f = fs[L::width() - 1];
        f(self, idxs, nm, fm)
    }

    pub fn sum_all_masked2_compr(&self, idxs: &BitVecRef, nm: &BitVecRef, fm: &BitVecRef) -> u64 {
        let m = fm.block_len::<u32>();
        assert_eq!(self.len(), idxs.len() * L::width());
        assert_eq!(self.len(), nm.len() * L::width());
        assert!(idxs.cast::<u32>().iter().all(|&i| (i as usize) < m));
        unsafe { self.sum_all_masked2_compr_unsafe(idxs, nm, fm) }
    }
}

impl <B, L> Deref for BitSlice<B, L>
where B: Borrow<[BitBlock]>,
      L: BitSliceLayout {
    type Target = [BitBlock];
    fn deref(&self) -> &[BitBlock] { self.vec.deref() }
}

impl <B, L> DerefMut for BitSlice<B, L>
where B: Borrow<[BitBlock]> + BorrowMut<[BitBlock]>,
      L: BitSliceLayout {
    fn deref_mut(&mut self) -> &mut [BitBlock] { self.vec.deref_mut() }
}


macro_rules! bitslice_info {
    ($name:ident, $width:expr) => {
        pub struct $name;
        impl BitSliceLayout for $name {
            #[inline(always)]
            fn width() -> usize { $width }
        }
    }
}

bitslice_info!(BitSliceLayout1, 1);
bitslice_info!(BitSliceLayout2, 2);
bitslice_info!(BitSliceLayout4, 4);






// ------------------------------------------------------------------------------------------------

#[cfg(test)]
mod test {
    use crate::NumT;
    use crate::bits::BitBlock;
    use crate::slice_store::{SliceStore, HistStore, BitBlockStore};
    use crate::slice_store::{BitSliceLayout};
    use crate::slice_store::{BitSliceLayout1, BitSliceLayout2, BitSliceLayout4};

    const BOUNDS: (NumT, NumT) = (0.0, 1.0);

    #[test]
    fn histograms() {
        let cardinalities = vec![4, 6, 16];
        let mut store = HistStore::<f32>::new(cardinalities.iter().cloned());
        let r0;
        let r1;
        let r2;
        let ptr0;
        let ptr1;
        let ptr2;

        {
            r0 = store.alloc_hists();

            assert_eq!(store.get_hist(r0, 0).len(), 4);
            assert_eq!(store.get_hist(r0, 1).len(), 6);
            assert_eq!(store.get_hist(r0, 2).len(), 16);

            ptr0 = &store.get_hist(r0, 0)[0] as *const f32;
        }

        {
            r1 = store.alloc_hists();
            store.free_hists(r0);
            r2 = store.alloc_hists();

            ptr1 = &store.get_hist(r1, 0)[0] as *const f32;
        }

        {
            ptr2 = &store.get_hist(r2, 0)[0] as *const f32;

            assert_ne!(ptr0, ptr1);
            assert_eq!(ptr0, ptr2); // histograms reused
        }
    }

    #[test]
    fn slicestore_aligned() {
        let elem_align = 16;
        let mut store = SliceStore::<f32>::aligned(128, elem_align * std::mem::size_of::<f32>());

        assert!(store.alloc_slice_default(1).0 as usize % elem_align == 0);
        assert!(store.alloc_slice_default(3).0 as usize % elem_align == 0);
        assert!(store.alloc_slice_default(31).0 as usize % elem_align == 0);
        assert!(store.alloc_slice_default(128).0 as usize % elem_align == 0);
    }

    #[test]
    fn slicestore_get_two_slices() {
        let mut store = SliceStore::<usize>::new(10);
        let r1 = store.alloc_slice_default(10);
        let r2 = store.alloc_slice_default(15);

        let (s1, s2) = store.get_two_slices_mut(r1, r2);
        for i in 0..10 { s1[i] = i + 50; }
        for j in 0..15 { s2[j] = j + 20; }

        let (s1, s2) = store.get_two_slices_mut(r1, r2);
        for i in 0..10 { assert_eq!(s1[i], i + 50); }
        for j in 0..15 { assert_eq!(s2[j], j + 20); }

        let (s2, s1) = store.get_two_slices_mut(r2, r1);
        for i in 0..10 { assert_eq!(s1[i], i + 50); }
        for j in 0..15 { assert_eq!(s2[j], j + 20); }

        let s1 = store.get_slice(r1);
        for i in 0..10 { assert_eq!(s1[i], i + 50); }

        let s2 = store.get_slice(r2);
        for j in 0..15 { assert_eq!(s2[j], j + 20); }
    }


    #[test]
    fn bitvec_basic_mut() {
        let n = 10_000;

        let mut store = BitBlockStore::new(16);
        let range = store.alloc_from_iter(n, 0..n);
        let mut bitvec = store.get_bitvec_mut(range);

        for i in 0..n {
            assert_eq!(bitvec.cast::<usize>()[i], i);
            bitvec.cast_mut::<usize>()[i] = 0;
            assert_eq!(bitvec.cast::<usize>()[i], 0);
            bitvec.cast_mut::<usize>()[i] = n - i;
        }

        for i in 0..n {
            assert_eq!(bitvec.cast::<usize>()[i], n-i);
        }
    }

    #[test]
    fn bitvec_from_bool_iter() {
        let n = 13456;
        let f = |k| k<n && k%13==1;
        let iter = (0..n).map(f);

        let mut store = BitBlockStore::new(16);
        let range = store.alloc_from_bits_iter(n, iter);
        let bitvec = store.get_bitvec(range);

        for (i, block) in bitvec.iter().enumerate() {
            for j in 0..BitBlock::nbits() {
                let k = i*BitBlock::nbits() + j;
                let b = f(k);
                assert_eq!(b, block.get_bit(j));
            }
        }
    }
    
    #[test]
    fn bitvec_one_bits() {
        let mut store = BitBlockStore::new(16);
        let range = store.alloc_one_bits(50);
        let bitvec = store.get_bitvec(range);

        assert_eq!(bitvec.get_bit(0), true);
        assert_eq!(bitvec.get_bit(49), true);
        assert_eq!(bitvec.get_bit(50), false);
        assert_eq!(bitvec.get_bit(51), false);
    }

    #[test]
    fn bitvec_from_iter() {
        let n = 4367;
        let f = |i| if i >= n as u32 { 0 } else { 101*i+13 };

        let mut store = BitBlockStore::new(16);
        let range = store.alloc_from_iter(n, (0u32..n as u32).map(f));
        let vec = store.get_bitvec(range);

        for (i, &b_u32) in vec.cast::<u32>().iter().enumerate() {
            assert_eq!(b_u32, f(i as u32));
        }

        for i in 0..n {
            assert_eq!(*vec.get::<u32>(i), f(i as u32));
        }

        for i in 0..n {
            unsafe {
                assert_eq!(*vec.get_unchecked::<u32>(i), f(i as u32));
            }
        }

        let mut vec = store.get_bitvec_mut(range);
        for i in 0..n { vec.set::<u32>(i, f(i as u32) + 10); }
        for i in 0..n { assert_eq!(*vec.get::<u32>(i), f(i as u32) + 10); }
    }

    #[test]
    fn bitvec_cast_len() {
        let n = 13456;
        let f = |k| k<n && k%31==1;
        let iter = (0..n).map(f);

        let mut store = BitBlockStore::new(16);
        let range = store.alloc_from_bits_iter(n, iter);
        let vec = store.get_bitvec(range);

        assert_eq!(vec.len(), n / 256 + 1);
        assert_eq!(vec.cast::<u128>().len(), vec.len() * 2);
        assert_eq!(vec.cast::<u64>().len(), vec.len() * 4);
        assert_eq!(vec.cast::<u32>().len(), vec.len() * 8);
        assert_eq!(vec.cast::<u16>().len(), vec.len() * 16);
        assert_eq!(vec.cast::<u8>().len(), vec.len() * 32);

        for (i, qword) in vec.cast::<u64>().iter().enumerate() {
            for j in 0..64 {
                let b = f(i*64 + j);
                assert_eq!(b, qword >> j & 0x1 == 0x1);
            }
        }
    }

    #[test]
    fn bitvec_alloc_zeros_end() {
        let mut store = BitBlockStore::new(16);
        {
            // allocate some memory
            let range = store.alloc_from_iter(3, 10u32..13u32);
            let v0 = store.get_bitvec(range);
            assert_eq!(v0.cast::<u32>()[1], 11);
            assert_eq!(v0.cast::<u32>().iter().cloned().last().unwrap(), 0);
        }

        for _ in 0..100 {
            {
                let range = store.alloc_from_iter(3, 10u32..13u32);
                let v1 = store.get_bitvec(range);
                for (i, &b_u32) in v1.cast::<u32>().iter().enumerate() {
                    if i < 3 { assert_eq!(b_u32, (10+i) as u32); }
                    else     { assert_eq!(b_u32, 0); }
                }
            }
            {
                let range = store.alloc_from_bits_iter(32, (0..32).map(|_| true));
                let v2 = store.get_bitvec(range);
                for (i, &b_u32) in v2.cast::<u32>().iter().enumerate() {
                    if i == 0 { assert_eq!(b_u32, 0xFFFFFFFF); }
                    else      { assert_eq!(b_u32, 0); }
                }
            }
        }
    }

    fn bitslice_get_set_value<I>()
    where I: BitSliceLayout {
        let n = 20_000;
        let scale = (I::nunique_values() - 1) as NumT;

        let mut store = BitBlockStore::new(16);
        let range = store.alloc_zero_bitslice::<I>(n);
        let mut slice = store.get_bitslice_mut::<I>(range);

        for i in 0..n {
            let k = ((101*i+37) % I::nunique_values() as usize) as u8;
            slice.set_value(i, k);
            assert_eq!(k, slice.get_value(i));

            slice.set_value(i, 0);
            assert_eq!(0, slice.get_value(i));

            let u = k as NumT / scale;
            slice.set_scaled_value(i, u, BOUNDS);
            //println!("bitslice_get_set_value: {} - {}", u, slice.get_scaled_value(i, BOUNDS));
            assert_eq!(u, slice.get_scaled_value(i, BOUNDS));
            assert_eq!(k, slice.get_value(i));
        }
    }

    fn bitslice_sum_block<L>()
    where L: BitSliceLayout {
        let n = 10_000;

        let mut store = BitBlockStore::new(16);
        let range = store.alloc_zero_bitslice::<L>(n);
        let mut slice = store.get_bitslice_mut::<L>(range);

        let mut sum_check = 0.0;
        let mut sum_check_u32 = 0u32;
        for i in 0..n {
            if i % 32 == 0 && i != 0 {
                let j = i / 32 - 1;
                let sum = slice.sum_scaled_masked(j, 0xFFFFFFFF, BOUNDS).0;
                let sum_u32 = slice.sum_masked(j, 0xFFFFFFFF);
                assert_eq!(sum_u32, sum_check_u32);
                assert!((sum - sum_check).abs() < 1e-5);
                sum_check = 0.0;
                sum_check_u32 = 0;
            }

            let k = (((101*i*i+37) >> 3) % L::nunique_values() as usize) as u8;
            slice.set_value(i, k);

            sum_check += slice.get_scaled_value(i, BOUNDS);
            sum_check_u32 += k as u32;
        }
    }

    fn bitslice_sum_all<L>()
    where L: BitSliceLayout {
        let n = 40_000;

        let mut sum_check1 = 0u64;
        let mut sum_check2 = 0u64;
        let mut store = BitBlockStore::new(16);
        let slice_r = store.alloc_zero_bitslice::<L>(n);
        let mask_r0 = store.alloc_one_bits(n);
        let mask_r1 = store.alloc_zero_bits(n);
        let mask_r2 = store.alloc_zero_bits(n);

        let mut slice = store.get_bitslice_mut::<L>(slice_r);
        for i in 0..n {
            let j = (6353*i*i+37) >> 1;
            let k = (j % L::nunique_values() as usize) as u8;
            slice.set_value(i, k);
            sum_check1 += k as u64;
        }

        let (mut mask1, mut mask2) = store.get_two_bitvecs_mut(mask_r1, mask_r2);
        for i in 0..n {
            let j = (6353*i*i+37) >> 1;
            let k = (j % L::nunique_values() as usize) as u8;
            mask1.set_bit(i, j%3==1);
            mask2.set_bit(i, j%6==1);
            sum_check2 += if mask1.get_bit(i) && mask2.get_bit(i) { k as u64 } else { 0 };
        }

        // sum uncompressed
        let slice = store.get_bitslice::<L>(slice_r);
        let mask0 = store.get_bitvec(mask_r0);
        let mask1 = store.get_bitvec(mask_r1);
        let mask2 = store.get_bitvec(mask_r2);

        let sum1 = slice.sum_all_masked2(&mask0, &mask0); // all 
        let sum2 = slice.sum_all_masked2(&mask1, &mask2); // selection 

        println!("bitslice_sum_all: {} - {}", sum_check1, sum1);
        println!("bitslice_sum_sel: {} - {}", sum_check2, sum2);
        assert_eq!(sum_check1, sum1);
        assert_eq!(sum_check2, sum2);
    }

    fn bitslice_sum_all_compr<L>()
    where L: BitSliceLayout {
        let n = 40_000;

        let mut sum_check = 0u64;
        let mut store = BitBlockStore::new(16);
        let slice_r = store.alloc_zero_bitslice::<L>(n);
        let mask_r1 = store.alloc_zero_bits(n);
        let n_u32 = store.get_bitvec(mask_r1).block_len::<u32>();
        let mask_r2 = store.alloc_zero_bits(4*n);
        let idxs_r = store.alloc_from_iter(n_u32, (0..n_u32).map(|i| (2*i) as u32));

        let mut slice = store.get_bitslice_mut::<L>(slice_r);
        for i in 0..n {
            let j = (6353*i+37) >> 1;
            let k = (j % L::nunique_values() as usize) as u8;
            slice.set_value(i, k);
        }

        let (mut mask1, mut mask2) = store.get_two_bitvecs_mut(mask_r1, mask_r2);
        for p in 0..n/32 {
            let pp = p*2;
            for q in 0..32 {
                let j = (6353*(p*32+q)+37) >> 1;
                let k = (j % L::nunique_values() as usize) as u8;
                mask1.set_bit(p *32 + q, j%3==1);
                mask2.set_bit(pp*32 + q, j%6==1);
                sum_check += if j%6==1 && j%3==1 { k as u64 } else { 0 };
            }
        }

        let slice = store.get_bitslice::<L>(slice_r);
        let mask1 = store.get_bitvec(mask_r1);
        let mask2 = store.get_bitvec(mask_r2);
        let idxs = store.get_bitvec(idxs_r);
        let sum = slice.sum_all_masked2_compr(&idxs, &mask1, &mask2);

        println!("bitslice_sum_all_compr: {} vs {}", sum_check, sum);
        assert_eq!(sum_check, sum);
    }

    #[test]
    fn bitslice() {
        bitslice_get_set_value::<BitSliceLayout1>();
        bitslice_sum_block::<BitSliceLayout1>();
        bitslice_sum_all::<BitSliceLayout1>();
        bitslice_sum_all_compr::<BitSliceLayout1>();

        bitslice_get_set_value::<BitSliceLayout2>();
        bitslice_sum_block::<BitSliceLayout2>();
        bitslice_sum_all::<BitSliceLayout2>();
        bitslice_sum_all_compr::<BitSliceLayout2>();

        bitslice_get_set_value::<BitSliceLayout4>();
        bitslice_sum_block::<BitSliceLayout4>();
        bitslice_sum_all::<BitSliceLayout4>();
        bitslice_sum_all_compr::<BitSliceLayout4>();
    }

    #[test]
    #[should_panic]
    fn bitslice_fail() {
        let mut store = BitBlockStore::new(16);
        let range = store.alloc_zero_bits(10);
        let _slice = store.get_bitslice::<BitSliceLayout4>(range);
    }

    #[test]
    fn bitvec_count_ones_and() {
        let n = 10_000;
        let mut all_equal = true;

        for i in 1..20 {
            let f1 = |k| k<n && k%i==1;
            let f2 = |k| k<n && k%(2*i)==1;

            let mut store = BitBlockStore::new(16);
            let r1 = store.alloc_from_bits_iter(n, (0..n).map(f1));
            let r2 = store.alloc_from_bits_iter(n, (0..n).map(f2));
            let v1 = store.get_bitvec(r1);
            let v2 = store.get_bitvec(r2);

            let mut count_ones_and = 0;
            for (i, (block1, block2)) in v1.iter().zip(v2.iter()).enumerate() {
                for j in 0..BitBlock::nbits() {
                    let k = i*BitBlock::nbits() + j;
                    let (b1, b2) = (f1(k), f2(k));
                    if b1 && b2 { count_ones_and += 1; }

                    assert_eq!(b1, block1.get_bit(j)); // sanity check
                    assert_eq!(b2, block2.get_bit(j));
                }
            }

            let block_count_ones_and = v1.count_ones_and(&v2);
            println!("count_ones_and_uc: {}, {}", count_ones_and, block_count_ones_and);
            all_equal &= count_ones_and == block_count_ones_and;
        }

        assert!(all_equal);
    }

    #[test]
    fn bitvec_count_ones_and_compr() {
        use crate::bits::bitblock::get_bit;

        let n = 10_000;
        let mut all_equal = true;

        for i in 1..20 {
            let f1 = |k| k<n && k%i==1;
            let f2 = |k| k<n && k%(2*i)==1;

            let mut store = BitBlockStore::new(16);
            let r1 = store.alloc_from_bits_iter(n, (0..n).map(f1));
            let r2 = store.alloc_from_bits_iter(n*3, (0..n*3).map(f2));
            let n_u32 = store.get_bitvec(r1).block_len::<u32>();
            let r3 = store.alloc_from_iter(n_u32, (0..n_u32).map(|i| (2*i) as u32));
            let v1 = store.get_bitvec(r1);
            let v2 = store.get_bitvec(r2);
            let idxs = store.get_bitvec(r3);

            let mut count_ones_and = 0;
            for i in 0..n_u32 {
                let block1 = *v1.get::<u32>(i);
                let block2 = *v2.get::<u32>(i*2);
                for j in 0..32 {
                    let k1 = i*32   + j;
                    let k2 = i*32*2 + j;
                    let (b1, b2) = (f1(k1), f2(k2));
                    if b1 && b2 { count_ones_and += 1; }

                    assert_eq!(b1, get_bit(block1, j as u8)); // sanity check
                    assert_eq!(b2, get_bit(block2, j as u8));
                }
            }

            let block_count_ones_and = v1.count_ones_and_compr(&idxs, &v2);
            println!("count_ones_and_c:  {}, {}", count_ones_and, block_count_ones_and);
            all_equal &= count_ones_and == block_count_ones_and;
        }

        assert!(all_equal);
    }
}