use bits::{BitSet, BitSlice};

pub struct ScaledBitSlice<T> {
    bitslice: BitSlice,
    lower_lim: T,
    upper_lim: T,
    len: usize,
}

fn min<T: PartialOrd + Copy>(a: T, b: T) -> T { if a.lt(&b) { a } else { b } }
fn max<T: PartialOrd + Copy>(a: T, b: T) -> T { if a.gt(&b) { a } else { b } }

macro_rules! scaled_bitslice_impl {
    ($t:ty) => {
        impl ScaledBitSlice<$t> {
            pub fn new<I>(len: usize, nbits: u8, mut iter: I, lower_lim: $t, upper_lim: $t) -> ScaledBitSlice<$t>
            where I: Iterator<Item = $t> {
                let mut slice = BitSlice::new(len, nbits);
                let mut count = 0;

                let d = upper_lim - lower_lim;
                let maxval = (slice.nunique_values() - 1) as $t;

                while let Some(x) = iter.next() {
                    let v0 = min(upper_lim, max(lower_lim, x));
                    let v1 = ((v0 - lower_lim) / d) * maxval;
                    let v2 = v1.round() as u8;
                    slice.set_value(count, v2);
                    count += 1;
                }

                ScaledBitSlice {
                    bitslice: slice,
                    lower_lim: lower_lim,
                    upper_lim: upper_lim,
                    len: len,
                }
            }

            fn linproj(&self, value: $t, n: $t) -> $t {
                let maxval = (self.bitslice.nunique_values() - 1) as $t;
                (value / maxval) * (self.upper_lim - self.lower_lim) + n * self.lower_lim
            }

            pub fn get_value(&self, index: usize) -> $t {
                self.linproj(self.bitslice.get_value(index) as $t, 1.0)
            }

            pub fn sum(&self) -> $t {
                let sum = self.bitslice.sum() as $t;
                self.linproj(sum, self.len as $t)
            }

            pub fn sum_masked(&self, mask: &BitSet) -> $t {
                let count = mask.true_count() as $t;
                let sum = self.bitslice.sum_masked(mask) as $t;
                self.linproj(sum, count)
            }

            /// Sum the elements for which (mask1 & mask2) is one; mask_one_count is the number of
            /// one bits in (mask1 & mask2), i.e. `mask1.count_and(mask2)`.
            pub fn sum_masked_and(&self, mask1: &BitSet, mask2: &BitSet, mask_one_count: u64) -> $t {
                let sum = self.bitslice.sum_masked_and(mask1, mask2) as $t;
                self.linproj(sum, mask_one_count as $t)
            }

            /// Sum the elements for which (mask1 & !mask2) is one; mask_one_count is the number of
            /// one bits in (mask1 & !mask2), i.e. `mask1.count_andnot(mask2)`.
            pub fn sum_masked_andnot(&self, mask1: &BitSet, mask2: &BitSet, mask_one_count: u64) -> $t {
                let sum = self.bitslice.sum_masked_andnot(mask1, mask2) as $t;
                self.linproj(sum, mask_one_count as $t)
            }

            pub fn mean(&self) -> $t {
                let sum = self.sum();
                sum / self.len as $t
            }

            pub fn mean_masked(&self, mask: &BitSet) -> $t {
                let count = mask.true_count() as $t;
                let sum = self.bitslice.sum_masked(mask) as $t;
                self.linproj(sum, count) / count
            }
        }
    }
}

// could also use num::Float, but NumCast::from sucks because Option and unwrap
scaled_bitslice_impl!(f32);
scaled_bitslice_impl!(f64);





#[cfg(test)]
mod test {
    use bits::BitSet;
    use bits::ScaledBitSlice;

    #[test]
    fn test_scaled_bitslice() {
        //              2    3     2     3    1    1      0     3
        let v = vec![0.25, 0.5, 0.25, 0.75, 0.0, 0.0, -0.25, 0.75];
        let v_capped = v.iter().map(|&x| f32::min(0.5, x)).collect::<Vec<f32>>();
        let sum_actual = v_capped.iter().sum();
        let target_values = ScaledBitSlice::<f32>::new(v.len(), 2, v.iter().cloned(), -0.25, 0.50);
        let sum = target_values.sum();
        let mean = target_values.mean();

        for (i, &x) in v_capped.iter().enumerate() {
            assert_eq!(target_values.get_value(i), x);
        }
        assert_eq!(sum, sum_actual);
        assert_eq!(mean, sum_actual / 8.0);

        let mask = BitSet::from_bool_iter(v.len(), vec![1,1,0,0,0,0,1,1].into_iter().map(|x| x==1));
        assert_eq!(target_values.sum_masked(&mask), 1.0);
        assert_eq!(target_values.mean_masked(&mask), 1.0 / 4.0);
        let mask = BitSet::from_bool_iter(v.len(), vec![1,1,0,0,0,0,1,1].into_iter().map(|x| x==0));
        assert_eq!(target_values.sum_masked(&mask), 0.75);
        assert_eq!(target_values.mean_masked(&mask), 0.75 / 4.0);
    }

    #[test]
    fn test_scaled_bitslice_sum_masked_and() {
        let n = 8;
        let v = vec![0.25, 0.5, 0.25, 0.50, 0.0, 0.0, -0.25, 0.50];
        let vs = ScaledBitSlice::<f32>::new(v.len(), 2, v.iter().cloned(), -0.25, 0.50);

        let m1 = BitSet::from_bool_iter(n,        vec![1,1,0,0,0,0,1,1].into_iter().map(|x| x==1));
        let m2 = BitSet::from_bool_iter(n,        vec![0,1,1,0,0,0,0,1].into_iter().map(|x| x==1));
        let m1and2 = BitSet::from_bool_iter(n,    vec![0,1,0,0,0,0,0,1].into_iter().map(|x| x==1));
        let m1andnot2 = BitSet::from_bool_iter(n, vec![1,0,0,0,0,0,1,0].into_iter().map(|x| x==1));

        assert_eq!(vs.sum_masked(&m1and2),      vs.sum_masked_and(&m1, &m2, 2));
        assert_eq!(vs.sum_masked(&m1andnot2),   vs.sum_masked_andnot(&m1, &m2, 2));
    }
}
