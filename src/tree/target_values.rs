use bits::{BitSet, BitSlice};
use dataset::NumericalType;

pub struct TargetValues {
    bitslice: BitSlice,
    lower_lim: NumericalType,
    upper_lim: NumericalType,
    len: usize,
}

impl TargetValues {
    pub fn new<I>(len: usize, nbits: u8, mut iter: I, lower_lim: NumericalType,
                  upper_lim: NumericalType) -> TargetValues
    where I: Iterator<Item = NumericalType>
    {
        let mut slice = BitSlice::new(len, nbits);
        let mut count = 0;

        let d = upper_lim - lower_lim;
        let maxval = (slice.nunique_values() - 1) as NumericalType;

        while let Some(x) = iter.next() {
            let v0 = NumericalType::min(upper_lim, NumericalType::max(lower_lim, x));
            let v1 = ((v0 - lower_lim) / d) * maxval;
            let v2 = v1.round() as u8;
            slice.set_value(count, v2);
            count += 1;
        }

        TargetValues {
            bitslice: slice,
            lower_lim: lower_lim,
            upper_lim: upper_lim,
            len: len,
        }
    }

    fn linproj(&self, value: NumericalType, n: NumericalType) -> NumericalType {
        let maxval = (self.bitslice.nunique_values() - 1) as NumericalType;
        (value / maxval) * (self.upper_lim - self.lower_lim) + n * self.lower_lim
    }

    pub fn get_value(&self, index: usize) -> NumericalType {
        self.linproj(self.bitslice.get_value(index) as NumericalType, 1 as NumericalType)
    }

    pub fn sum(&self) -> NumericalType {
        let sum = self.bitslice.sum() as NumericalType;
        self.linproj(sum, self.len as NumericalType)
    }

    pub fn sum_masked(&self, mask: &BitSet) -> NumericalType {
        let count = mask.count_ones() as NumericalType;
        let sum = self.bitslice.sum_masked(mask) as NumericalType;
        self.linproj(sum, count)
    }
}






#[cfg(test)]
mod test {
    use bits::BitSet;
    use tree::TargetValues;
    use dataset::NumericalType;

    #[test]
    fn test_target_values() {
        //              2    3     2     3    1    1      0     3
        let v = vec![0.25, 0.5, 0.25, 0.75, 0.0, 0.0, -0.25, 0.75];
        let v_capped = v.iter().map(|&x| NumericalType::min(0.5, x)).collect::<Vec<NumericalType>>();
        let sum_actual = v_capped.iter().sum();
        let target_values = TargetValues::new(v.len(), 2, v.iter().map(|&x| x), -0.25, 0.50);
        let sum = target_values.sum();

        for (i, &x) in v_capped.iter().enumerate() {
            assert_eq!(target_values.get_value(i), x);
        }
        assert_eq!(sum, sum_actual);

        let mask = BitSet::from_bool_iter(vec![1,1,0,0,0,0,1,1].into_iter().map(|x| x==1));
        assert_eq!(target_values.sum_masked(&mask), 1.0);
        let mask = BitSet::from_bool_iter(vec![1,1,0,0,0,0,1,1].into_iter().map(|x| x==0));
        assert_eq!(target_values.sum_masked(&mask), 0.75);
    }
}
