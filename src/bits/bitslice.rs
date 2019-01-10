use std::fmt::{Debug, Formatter, Result as FmtResult};
use std::marker::PhantomData;

use NumT;

use bits::{BitBlock, BitVec};
use bits::bitblock::{get_bit, set_bit, get_blockpos, get_bitpos};

pub trait BitSliceLayout: Clone {
    /// Number of bit lines in the bit slice.
    fn width() -> usize;

    /// The order of the lanes.
    fn order(i: usize) -> u64;
     
    /// The number of times the `order` pattern needs to be repeated to fill up 256 bits.
    fn nrepeats() -> usize { 8 / Self::width() }
}

macro_rules! impl_bitslice_width {
    ($name:ident, $order:expr) => {
        #[derive(Clone)]
        pub struct $name;
        impl BitSliceLayout for $name {
            fn width() -> usize { $order.len() }
            fn order(i: usize) -> u64 {
                let w = $order;
                debug_assert!(i < w.len());
                unsafe { *w.get_unchecked(i) }
            }
        }
    }
}

const LAYOUT_ORDER1: [u64; 1] = [1];
const LAYOUT_ORDER2: [u64; 2] = [1, 2];
const LAYOUT_ORDER4: [u64; 4] = [1, 2, 4, 8];

impl_bitslice_width!(BitSliceLayout1, LAYOUT_ORDER1);
impl_bitslice_width!(BitSliceLayout2, LAYOUT_ORDER2);
impl_bitslice_width!(BitSliceLayout4, LAYOUT_ORDER4);



// ------------------------------------------------------------------------------------------------

#[derive(Clone)]
pub struct BitSlice<L: BitSliceLayout> {
    vec: BitVec,
    _marker: PhantomData<L>,
}

impl <L> BitSlice<L>
where L: BitSliceLayout {
    pub fn new(nbits: usize) -> BitSlice<L> {
        let nblocks = BitBlock::blocks_required_for(nbits);
        let vec = BitVec::zero_blocks(nblocks * L::width());

        BitSlice {
            vec: vec,
            _marker: PhantomData,
        }
    }

    pub fn random(nbits: usize) -> BitSlice<L> {
        use rand::prelude::*;
        use rand::distributions::Uniform;

        let mask = (1 << L::width()) - 1;
        let mut rng = thread_rng();
        let dist = Uniform::new(0u8, 0xFF);
        let mut slice = BitSlice::new(nbits);

        for i in 0..nbits {
            slice.set_value(i, dist.sample(&mut rng) & mask);
        }

        slice
    }

    pub fn copy_blocks32(other: &BitSlice<L>, indices: &[u32]) -> BitSlice<L> {
        let mut slice = BitSlice::new(indices.len() * 32);
        {
            let svec = slice.vec.cast_mut::<u32>();
            let ovec = other.vec.cast::<u32>();

            for (i, &ju32) in indices.iter().enumerate() {
                let j = ju32 as usize;
                for k in 0..L::width() {
                    svec[i * L::width() + k] = ovec[j * L::width() + k];
                }
            }
        }
        slice
    }

    pub fn get_value(&self, index: usize) -> u8 {
        let mut res = 0;
        let i = get_blockpos::<u32>(index);
        let j = get_bitpos::<u32>(index);
        let vec_u32 = self.vec.cast::<u32>();
        for k in 0..L::width() {
            let bits = vec_u32[i * L::width() + k];
            let b = get_bit(bits, j);
            res = set_bit(res, k as u8, b);
        }
        res
    }

    pub fn set_value(&mut self, index: usize, value: u8) {
        let i = get_blockpos::<u32>(index);
        let j = get_bitpos::<u32>(index);
        let vec_u32 = self.vec.cast_mut::<u32>();
        for k in 0..L::width() {
            let bits = &mut vec_u32[i * L::width() + k];
            let b = get_bit(value, k as u8);
            *bits = set_bit(*bits, j, b);
        }
    }

    pub fn nblocks(&self) -> usize { self.vec.nblocks() }
    pub fn nbytes(&self) -> usize { self.vec.nbytes() }
    pub fn nbits(&self) -> usize { self.vec.nbits() }
    pub fn width(&self) -> u8 { L::width() as u8 }
    pub fn nunique_values(&self) -> u8 { 1 << L::width() }

    /// Sum a 32bit block
    pub fn sum_block32(&self, blockpos: usize, mask: u32) -> u64 {
        let vec_u32 = self.vec.cast::<u32>();
        let mut sum = 0;
        for k in 0..L::width() {
            let bits = vec_u32[blockpos + k] & mask;
            sum += bits.count_ones() as u64 * L::order(k);
        }
        sum
    }
}






// ------------------------------------------------------------------------------------------------

#[derive(Clone)]
pub struct ScaledBitSlice<L: BitSliceLayout> {
    bitslice: BitSlice<L>, 
    lo: NumT,
    up: NumT,
    nvalues: usize,
}

impl <L: BitSliceLayout> ScaledBitSlice<L> {
    pub fn new<I>(nvalues: usize, lo: NumT, up: NumT, mut iter: I) -> Self
    where I: Iterator<Item = NumT> {
        let mut slice = BitSlice::<L>::new(nvalues);
        let mut count = 0;

        let d = up - lo;
        let maxval = (slice.nunique_values() - 1) as NumT;

        while let Some(x) = iter.next() {
            let v0 = NumT::min(up, NumT::max(lo, x));
            let v1 = ((v0 - lo) / d) * maxval;
            let v2 = v1.round() as u8;
            //println!("{}, {}", x, v2);
            slice.set_value(count, v2);
            count += 1;
        }

        ScaledBitSlice {
            bitslice: slice,
            lo: lo,
            up: up,
            nvalues: nvalues,
        }
    }

    pub fn from_bitslice(nvalues: usize, slice: BitSlice<L>, lo: NumT, up: NumT) -> Self {
        ScaledBitSlice {
            bitslice: slice,
            lo: lo,
            up: up,
            nvalues: nvalues,
        }
    }

    pub fn copy_blocks32(other: &ScaledBitSlice<L>, indices: &[u32]) -> ScaledBitSlice<L> {
        let slice = BitSlice::copy_blocks32(&other.bitslice, indices);
        ScaledBitSlice::from_bitslice(indices.len() * 32, slice, other.lo, other.up)
    }

    pub fn random(nvalues: usize, lo: NumT, up: NumT) -> Self {
        let slice = BitSlice::<L>::random(nvalues);

        ScaledBitSlice {
            bitslice: slice,
            lo: lo,
            up: up,
            nvalues: nvalues,
        }
    }

    fn linproj(&self, value: NumT, n: NumT) -> NumT {
        let maxval = (self.bitslice.nunique_values() - 1) as NumT;
        (value / maxval) * (self.up - self.lo) + n * self.lo
    }

    pub fn get_value(&self, index: usize) -> NumT {
        self.linproj(self.bitslice.get_value(index) as NumT, 1.0)
    }

    pub fn nvalues(&self) -> usize {
        self.nvalues
    }

    /// (mask one count, sum)
    pub fn sum_block32(&self, index: usize, mask: u32) -> (u32, NumT) {
        let sum = self.bitslice.sum_block32(index, mask) as NumT;
        let count = mask.count_ones();
        (count, self.linproj(sum, count as NumT))
    }
}

impl <L: BitSliceLayout> Debug for ScaledBitSlice<L> {
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        writeln!(f, "ScaledBitSlice with {} values", self.nvalues)?;
        for i in 0..self.nvalues {
            writeln!(f, "{:4}: {}", i, self.get_value(i))?;
        }
        Ok(())
    }
}






// ------------------------------------------------------------------------------------------------

#[cfg(test)]
mod test {
    use NumT;

    use bits::{BitSlice, BitSliceLayout};
    use bits::bitslice::{BitSliceLayout1, BitSliceLayout2, BitSliceLayout4};
    use bits::ScaledBitSlice;

    fn get_set_value<L: BitSliceLayout>() {
        let n = 10_000;
        let mut slice = BitSlice::<L>::new(n);
        for i in 0..n {
            let k = ((101*i+37) % slice.nunique_values() as usize) as u8;
            slice.set_value(i, k);
            assert_eq!(k, slice.get_value(i));
        }
    }

    #[test]
    fn bitslice_basic() {
        get_set_value::<BitSliceLayout1>();
        get_set_value::<BitSliceLayout2>();
        get_set_value::<BitSliceLayout4>();
    }

    fn get_value_bitslice<L: BitSliceLayout>() {
        let n = 10_000;
        let mut slice = BitSlice::<L>::new(n);
        let nunique = slice.nunique_values();

        for i in 0..n {
            let k = ((101*i+37) % nunique as usize) as u8;
            slice.set_value(i, k);
        }

        let scaled = ScaledBitSlice::<L>::from_bitslice(n, slice, 0.0, 1.0);

        for i in 0..n {
            let k = ((101*i+37) % nunique as usize) as u8;
            let u = k as NumT / (nunique-1) as NumT;
            assert_eq!(scaled.get_value(i), u);
        }
    }

    #[test]
    fn scaled_bitslice_basic() {
        get_value_bitslice::<BitSliceLayout1>();
        get_value_bitslice::<BitSliceLayout2>();
        get_value_bitslice::<BitSliceLayout4>();
    }

    fn bitset_sum_block32<L: BitSliceLayout>() {
        let n = 10_000;
        let mut slice = BitSlice::<L>::new(n);
        let nunique = slice.nunique_values();
        let f = |i| ((23*i + 1245) % nunique as usize) as u8;

        for i in 0..n {
            slice.set_value(i, f(i));
        }

        let mut sum = 0;
        for i in 0..n {
            if i % 32 == 0 && i > 0 {
                assert_eq!(sum, slice.sum_block32((i/32)-1, 0xFFFF));
                sum = 0;
            }

            if i % 32 < 16 {
                sum += f(i) as u64;
            }
        }
    }

    #[test]
    fn bitset_sum_block32_test() {
        bitset_sum_block32::<BitSliceLayout1>();
        bitset_sum_block32::<BitSliceLayout2>();
        bitset_sum_block32::<BitSliceLayout4>();
    }

    fn bitslice_sum_block32<L: BitSliceLayout>() {
        let n = 10_000;
        let mut slice = BitSlice::<L>::new(n);
        let nunique = slice.nunique_values();
        let f = |i| ((101*i+137) % nunique as usize) as u8;

        for i in 0..n {
            slice.set_value(i, f(i));
        }

        let scaled = ScaledBitSlice::<L>::from_bitslice(n, slice, 0.0, 1.0);

        let mut sum = 0.0;
        for i in 0..n {
            if i % 32 == 0 && i > 0 {
                assert_eq!(sum, scaled.sum_block32((i/32)-1, 0xFFFF).1);
                sum = 0.0;
            }

            if i % 32 < 16 {
                let u = f(i) as NumT / (nunique-1) as NumT;
                sum += u;
            }
        }
    }

    #[test]
    fn bitslice_sum_block32_test() {
        bitslice_sum_block32::<BitSliceLayout1>();
        bitslice_sum_block32::<BitSliceLayout2>();
        bitslice_sum_block32::<BitSliceLayout4>();
    }

    fn scaled_bitslice_copy<L: BitSliceLayout>() {
        let n = 10_000;
        let mut slice = BitSlice::<L>::new(n);
        let nunique = slice.nunique_values();

        for i in 0..n {
            let k = (((101*i+37) >> 2) % nunique as usize) as u8;
            slice.set_value(i, k);
        }

        let scaled = ScaledBitSlice::<L>::from_bitslice(n, slice, 0.0, 1.0);
        let indices = (0..(n/32) as u32).filter(|&i| i%3==0).collect::<Vec<u32>>();
        let copy = ScaledBitSlice::<L>::copy_blocks32(&scaled, &indices);

        for (i, &ju32) in indices.iter().enumerate() {
            let j = ju32 as usize;
            for k in 0..32 {
                let a = j*32+k;
                let b = i*32+k;
                //println!("j={}: {}, i={}: {}", a, scaled.get_value(a), b, copy.get_value(b));
                assert_eq!(scaled.get_value(a), copy.get_value(b));
            }
        }
    }

    #[test]
    fn scaled_bitslice_copy_test() {
        scaled_bitslice_copy::<BitSliceLayout1>();
        scaled_bitslice_copy::<BitSliceLayout2>();
        scaled_bitslice_copy::<BitSliceLayout4>();
    }
}

