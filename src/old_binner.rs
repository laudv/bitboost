use std::ops::Add;

use crate::NumT;


pub fn get_linpol_delta((min_value, max_value): (NumT, NumT), nbins: usize) -> NumT {
    (max_value - min_value) / (nbins - 1) as NumT
}

pub fn linpol((min_value, max_value): (NumT, NumT), bin: usize, nbins: usize) -> NumT {
    let delta = get_linpol_delta((min_value, max_value), nbins);
    linpol_delta(min_value, delta, bin, nbins)
}

pub fn linpol_delta(min_value: NumT, delta: NumT, bin: usize, nbins: usize) -> NumT {
    min_value + bin.min(nbins - 1) as NumT * delta
}

// TODO remove deprecated
pub struct Binner<'a, T, Comb>
where T: Clone,
      Comb: Fn(&mut T, T),
{
    count: u32,
    bins: &'a mut [T],
    combiner: Comb,
    min_value: NumT,
    delta: NumT,
}

impl <'a, T, Comb> Binner<'a, T, Comb>
where T: Clone, 
      Comb: Fn(&mut T, T),
{
    pub fn new(bins: &'a mut [T], (min_value, max_value): (NumT, NumT), combiner: Comb) -> Self {
        assert!(min_value.is_finite());
        assert!(max_value.is_finite());
        assert!(min_value < max_value);
        assert!(!bins.is_empty());

        let delta = get_linpol_delta((min_value, max_value), bins.len());

        Binner {
            count: 0,
            bins,
            combiner,
            min_value,
            delta,
        }
    }

    pub fn len(&self) -> usize { self.bins.len() }
    pub fn count(&self) -> u32 { self.count }

    fn get_bin(&self, value: NumT) -> usize {
        let x = (value - self.min_value) / self.delta;
        let i = x.floor() as isize;
        (i.max(0) as usize).min(self.len() - 1)
    }

    pub fn insert(&mut self, value: NumT, data: T) {
        let bin = self.get_bin(value);
        (self.combiner)(&mut self.bins[bin], data);
        self.count += 1;
    }

    pub fn bin_representative(&self, bin: usize) -> NumT {
        linpol_delta(self.min_value, self.delta, bin, self.len())
    }

    pub fn bin_with_rank<I, F>(&self, rank: I, f: F) -> (usize, I, I)
    where I: Copy + Ord + Add<Output=I> + Default,
          F: Fn(&T) -> I,
    {
        let mut accum = I::default();
        let mut prev_accum = I::default();
        let mut bin = 0;
        for b in self.bins.iter() {
            let x = f(b);
            bin += 1;
            prev_accum = accum;
            accum = accum + x;
            if accum > rank { break; }
        }

        debug_assert!(bin <= self.len());

        (bin - 1, prev_accum, accum)
    }
}
