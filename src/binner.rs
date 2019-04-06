/*
 * Copyright (c) DTAI - KU Leuven – All rights reserved.
 * Proprietary, do not copy or distribute without permission.
 * Written by Laurens Devos, 2019
*/

use std::ops::AddAssign;

use crate::NumT;

pub struct Binner<'a, BinT> {
    bins: &'a mut [BinT],
    min_value: NumT,
    delta: NumT,
}

impl <'a, BinT> Binner<'a, BinT> {
    pub fn new(bins: &'a mut [BinT], limits: (NumT, NumT)) -> Binner<'a, BinT> {
        assert!(limits.0.is_finite());
        assert!(limits.1.is_finite());
        assert!(limits.0 < limits.1);
        assert!(!bins.is_empty());

        let delta = (limits.1 - limits.0) / bins.len() as NumT;

        Binner {
            bins,
            min_value: limits.0,
            delta,
        }
    }

    pub fn insert<D, F>(&mut self, value: NumT, data: D, combiner: F)
    where F: Fn(&mut BinT, D) {
        let bin_index = self.get_bin(value);
        let bin = &mut self.bins[bin_index];
        (combiner)(bin, data);
    }

    pub fn bin_with_rank<F>(&self, rank: u32, extractor: F) -> usize
    where F: Fn(&BinT) -> u32,
    {
        let ranks = [rank];
        let iter = (&ranks).iter().cloned();
        self.rank_iter(iter, extractor).next().unwrap()
    }

    pub fn rank_iter<'b, R, Iter, F>(&'b self, ranks: Iter, extractor: F)
        -> RankIter<'b, R, BinT, Iter, F>
    where R: Copy + Default + PartialOrd + AddAssign,
          Iter: Iterator<Item=R>,
          F: Fn(&BinT) -> R,
    {
        RankIter {
            bins: self.bins,
            ranks,
            extractor,
            accum: Default::default(),
            prev_accum: Default::default(),
            bin_index: 0,
        }
    }

    pub fn bin_representative(&self, bin: usize) -> NumT {
        self.min_value + bin.min(self.bins.len() - 1) as NumT * self.delta
    }

    pub fn get_bin(&self, value: NumT) -> usize {
        let x = (value - self.min_value) / self.delta;
        let i = x.floor() as isize;
        (i.max(0) as usize).min(self.bins.len() - 1)
    }

    pub fn bins_mut(&mut self) -> &mut [BinT] {
        self.bins
    }

    pub fn bin_value(&self, index: usize) -> BinT
    where BinT: Copy {
        self.bins[index]
    }
}

pub struct RankIter<'a, R, BinT, Iter, F>
where Iter: Iterator<Item=R>,
      F: Fn(&BinT) -> R,
{
    bins: &'a [BinT],
    ranks: Iter,
    extractor: F,
    accum: R,
    prev_accum: R,
    bin_index: usize,
}

impl <'a, R, BinT, Iter, F> Iterator for RankIter<'a, R, BinT, Iter, F>
where R: Copy + PartialOrd + Default + AddAssign + std::fmt::Display,
      Iter: Iterator<Item=R>,
      F: Fn(&BinT) -> R,
{
    type Item = usize;
    fn next(&mut self) -> Option<Self::Item> {
        let rank = self.ranks.next();
        if rank.is_none() { return None; }
        let rank = rank.unwrap();

        debug_assert!(rank >= Default::default());

        while self.accum <= rank && self.bin_index < self.bins.len() {
            let x = (self.extractor)(&self.bins[self.bin_index]);
            self.bin_index += 1;
            self.prev_accum = self.accum;
            self.accum += x;
        }

        debug_assert!(self.accum >= rank);
        debug_assert!(self.bin_index <= self.bins.len());

        Some(self.bin_index - 1)
    }
}

impl <'a, R, BinT, Iter, F> RankIter<'a, R, BinT, Iter, F>
where R: Copy + PartialOrd + AddAssign,
      Iter: Iterator<Item=R>,
      F: Fn(&BinT) -> R,
{
    pub fn accum(&self) -> (R, R) {
        (self.prev_accum, self.accum)
    }
}








#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn binner_basic() {
        let mut v = vec![0u32; 4];
        let mut b = Binner::new(&mut v, (0.0, 1.0));

        assert_eq!(b.delta, 0.25);

        let combiner = |bin: &mut u32, _: ()| *bin += 1;
        let extractor = |bin: &u32| *bin;

        b.insert(-0.01,  (), combiner); // bin 0
        b.insert(0.00,   (), combiner);
        b.insert(0.0001, (), combiner);
        b.insert(0.2499, (), combiner);
        b.insert(0.25,   (), combiner); // bin 1
        b.insert(0.4999, (), combiner);
        b.insert(0.50,   (), combiner); // bin 2
        b.insert(0.7499, (), combiner);
        b.insert(0.75,   (), combiner); // bin 3
        b.insert(0.9999, (), combiner);
        b.insert(1.00,   (), combiner);
        b.insert(1.01,   (), combiner);

        assert_eq!(b.bin_representative(0), 0.00);
        assert_eq!(b.bin_representative(1), 0.25);
        assert_eq!(b.bin_representative(2), 0.50);
        assert_eq!(b.bin_representative(3), 0.75);

        let bins = vec![0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 3, 3];
        for i in 0..=11 {
            assert_eq!(b.bin_with_rank(i as u32, extractor), bins[i]);
        }

        let ranks = (0..=11).collect::<Vec<u32>>();
        let rank_iter = b.rank_iter(ranks.iter().cloned(), extractor);
        for (i, bin) in rank_iter.enumerate() {
            assert_eq!(bin, bins[i]);
        }

        assert_eq!(&v, &[4, 2, 2, 4]);
    }
}
