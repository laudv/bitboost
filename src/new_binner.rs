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

    pub fn bin_with_rank<F>(&self, rank: u32, extractor: F) -> (usize, u32, u32)
    where F: Fn(&BinT) -> u32,
    {
        //let mut accum = I::default();
        //let mut prev_accum = I::default();
        //let mut bin = 0;
        //for b in self.bins.iter() {
        //    let x = (extractor)(b);
        //    bin += 1;
        //    prev_accum = accum;
        //    accum = accum + x;
        //    if accum > rank { break; }
        //}

        //debug_assert!(bin <= self.bins.len());

        //(bin - 1, prev_accum, accum)
        
        let ranks = [rank];
        let iter = (&ranks).iter().cloned();
        self.rank_iter(iter, extractor).next().unwrap()
    }

    pub fn rank_iter<'b, Iter, F>(&'b self, ranks: Iter, extractor: F)
        -> RankIter<'b, BinT, Iter, F>
    where Iter: Iterator<Item=u32>,
          F: Fn(&BinT) -> u32,
    {
        RankIter {
            bins: self.bins,
            ranks,
            extractor,
            accum: 0,
            prev_accum: 0,
            bin_index: 0,
        }
    }

    /// Iterate over all bins containing the items with the given quantiles. Count is the total
    /// number of elements that are considered to be in this structure.
    /// TODO remove
    pub fn quantile_iter<'b, Iter, F>(&'b self, quantiles: Iter, count: usize,
                                               extractor: F)
        -> impl Iterator<Item = (usize, u32, u32)> + 'b
    where Iter: Iterator<Item=NumT> + 'b,
          F: Fn(&BinT) -> u32 + 'b,
    {
        let count = count as NumT;
        let ranks = quantiles.map(move |q| (q.max(0.0).min(1.0) * count).round() as u32);
        self.rank_iter(ranks, extractor)
    }

    pub fn bin_representative(&self, bin: usize) -> NumT {
        self.min_value + bin.min(self.bins.len() - 1) as NumT * self.delta
    }

    pub fn get_bin(&self, value: NumT) -> usize {
        let x = (value - self.min_value) / self.delta;
        let i = x.floor() as isize;
        (i.max(0) as usize).min(self.bins.len() - 1)
    }
}

pub struct RankIter<'a, BinT, Iter, F>
where Iter: Iterator<Item=u32>,
      F: Fn(&BinT) -> u32,
{
    bins: &'a [BinT],
    ranks: Iter,
    extractor: F,
    accum: u32,
    prev_accum: u32,
    bin_index: usize,
}

impl <'a, BinT, Iter, F> Iterator for RankIter<'a, BinT, Iter, F>
where Iter: Iterator<Item=u32>,
      F: Fn(&BinT) -> u32,
{
    type Item = (usize, u32, u32);
    fn next(&mut self) -> Option<Self::Item> {
        let rank = self.ranks.next();
        if rank.is_none() { return None; }
        let rank = rank.unwrap();

        println!("rank = {}", rank);

        while self.accum <= rank && self.bin_index < self.bins.len() {
            let x = (self.extractor)(&self.bins[self.bin_index]);
            self.bin_index += 1;
            self.prev_accum = self.accum;
            self.accum += x;
        }

        debug_assert!(self.accum >= rank);
        debug_assert!(self.bin_index <= self.bins.len());
        Some((self.bin_index - 1, self.prev_accum, self.accum))
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

        let rank_tuples = vec![(0, 0, 4), (0, 0, 4), (0, 0, 4), (0, 0, 4), (1, 4, 6), (1, 4, 6),
            (2, 6, 8), (2, 6, 8), (3, 8, 12), (3, 8, 12), (3, 8, 12), (3, 8, 12)];

        for i in 0..=11 {
            assert_eq!(b.bin_with_rank(i as u32, extractor), rank_tuples[i]);
        }

        let ranks = (0..=11).collect::<Vec<u32>>();
        let bins = b.rank_iter(ranks.iter().cloned(), extractor);
        for (i, bin) in bins.enumerate() {
            assert_eq!(bin, rank_tuples[i]);
        }

        let quantiles = (0..=11).map(|i| i as NumT / 12.0).collect::<Vec<NumT>>();
        let bins = b.quantile_iter(quantiles.iter().cloned(), 12, extractor);
        for (i, bin) in bins.enumerate() {
            assert_eq!(bin, rank_tuples[i]);
        }

        assert_eq!(&v, &[4, 2, 2, 4]);
    }
}
