use crate::NumT;

pub struct Histogram<D>
where D: Default {
    min_value: NumT,
    max_value: NumT,
    delta: NumT,
    count: u32,

    buckets: Vec<(u32, D)>, // count, data
    cumsum: Vec<u32>,
}

impl <D> Histogram<D>
where D: Default {
    pub fn new(nbuckets: usize) -> Histogram<D> {
        let buckets = (0..nbuckets).map(|_| Default::default()).collect();
        let cumsum = vec![0; nbuckets];
        Histogram {
            min_value: 0.0,
            max_value: 0.0,
            delta: 0.0,
            count: 0,
            buckets,
            cumsum,
        }
    }

    pub fn set_min_max(&mut self, min_value: NumT, max_value: NumT) {
        self.min_value = min_value;
        self.max_value = max_value;
        self.delta = (max_value - min_value) / self.buckets.len() as NumT;
    }

    fn get_bucket(&self, value: NumT) -> usize {
        debug_assert!(self.min_value != self.max_value);
        debug_assert!(self.delta > 0.0);

        let x = (value - self.min_value).max(0.0);
        let x = x.min(self.max_value - self.min_value) / self.delta;
        let i = x.floor() as usize;

        i
    }

    fn get_bucket_representative(&self, bucket: usize) -> NumT {
        self.min_value + bucket as NumT * self.delta
    }

    /// For repeated rank queries, use `cumsum`.
    fn get_bucket_with_rank(&self, rank: usize) -> (usize, u32) {
        assert!(rank < u32::max_value() as usize);
        let rank = rank as u32;

        let mut count_accum = 0;
        let mut bucket = 0;
        for b in &self.buckets {
            let count = b.0;
            if count_accum + count > rank { break; }
            count_accum += count;
            bucket += 1;
        }

        (bucket, count_accum) // bucket index, accumulated element count
    }

    fn quantile_in_bucket(&self, bucket: usize, cumsum: u32, rank: usize) -> NumT {
        let bucket_count = self.buckets[bucket].0;
        debug_assert_ne!(bucket_count, 0);
        let diff = (1 + rank - cumsum as usize) as NumT;
        diff / bucket_count as NumT
    }

    fn rank_from_quantile(&self, quantile: NumT) -> usize {
        (quantile * (self.count - 1) as NumT).round() as usize
    }

    pub fn cumsum<'a>(&'a mut self) -> CumSum<'a, D> {
        let mut count_accum = 0;
        for i in 0..self.buckets.len() {
            count_accum += self.buckets[i].0;
            self.cumsum[i] = count_accum;
        }

        CumSum {
            hist: self,
        }
    }

    pub fn reset(&mut self) {
        self.buckets.iter_mut().for_each(|x| *x = Default::default());
    }
}

impl Histogram<()> {
    pub fn insert(&mut self, value: NumT) {
        let bucket = self.get_bucket(value);
        self.buckets[bucket].0 += 1;
        self.count += 1;
    }

    fn approx_quantile_aux(&self, bucket: usize, cumsum: u32, rank: usize) -> NumT {
        let u = self.quantile_in_bucket(bucket, cumsum, rank);
        let v1 = self.get_bucket_representative(bucket);
        let v2 = self.get_bucket_representative(bucket + 1);
        println!("bucket: {} -> u={:.3}, v1={:.3}, v2={:.3}", bucket, u, v1, v2);
        (1.0 - u) * v1 + u * v2
    }

    pub fn approx_quantile(&self, quantile: NumT) -> NumT {
        let rank = self.rank_from_quantile(quantile);
        let (bucket, cumsum) = self.get_bucket_with_rank(rank);
        self.approx_quantile_aux(bucket, cumsum, rank)
    }
}

impl Histogram<NumT> {
    pub fn insert(&mut self, value: NumT) {
        let bucket = self.get_bucket(value);
        let b = &mut self.buckets[bucket];
        b.0 += 1;
        b.1 += value;
        self.count += 1;
    }
}


pub struct CumSum<'a, D>
where D: Default {
    hist: &'a Histogram<D>
}

impl <'a, D> CumSum<'a, D>
where D: Default {
    fn get_bucket_with_rank(&self, rank: usize) -> usize {
        assert!(rank < u32::max_value() as usize);
        let rank = rank as u32;

        match self.hist.cumsum.binary_search(&rank) {
            Ok(bucket)  => bucket+1,
            Err(bucket) => bucket,
        }
    }
}

impl <'a> CumSum<'a, ()> {
    pub fn approx_quantile(&self, quantile: NumT) -> NumT {
        let rank = self.hist.rank_from_quantile(quantile);
        let bucket = self.get_bucket_with_rank(rank);
        let cumsum = self.hist.cumsum[bucket] - self.hist.buckets[bucket].0;
        self.hist.approx_quantile_aux(bucket, cumsum, rank)
    }
}

impl <'a> CumSum<'a, NumT> {
    pub fn approx_quantile(&self, quantile: NumT) -> NumT {
        let rank = self.hist.rank_from_quantile(quantile);
        let bucket =self.get_bucket_with_rank(rank);
        let b = self.hist.buckets[bucket];
        b.1 / b.0 as NumT // mean of accumulated values
    }
}




#[cfg(test)]
mod test {
    use crate::NumT;
    use super::Histogram;

    #[test]
    fn rank() {
        let n = 10_000;
        let nbuckets = 100;
        let mut hist = Histogram::<()>::new(nbuckets);
        hist.set_min_max(0.0, 1.0);
        (0..n).map(|i| i as NumT / n as NumT).for_each(|x| hist.insert(x));

        for i in 0..nbuckets {
            assert_eq!(hist.buckets[i].0 as usize, n / nbuckets);
        }

        // Correct results for get_bucket_with_rank?
        for i in 0..n {
            let bucket = hist.get_bucket_with_rank(i).0;
            let expected = i / (n / nbuckets);
            assert_eq!(bucket, expected);
        }

        // Does cumsum give same results?
        let buckets = (0..n).map(|i| hist.get_bucket_with_rank(i).0).collect::<Vec<usize>>();
        let cumsum = hist.cumsum();
        for i in 0..n {
            let bucket1 = cumsum.get_bucket_with_rank(i);
            let bucket2 = buckets[i];
            assert_eq!(bucket1, bucket2);
        }
    }

    #[test]
    fn representative() {
        let nbuckets = 100;
        let mut hist = Histogram::<()>::new(nbuckets);
        hist.set_min_max(1.0, 2.0);

        for i in 0..nbuckets+1 {
            let v = hist.get_bucket_representative(i);
            assert!((1.0 + i as NumT / nbuckets as NumT - v).abs() < 1e-6);
        }
    }

    #[test]
    fn quantile_in_bucket() {
        let n = 100;
        let nbuckets = 5;
        let mut hist = Histogram::<()>::new(nbuckets);
        hist.set_min_max(0.0, 1.0);
        (0..n).map(|i| i as NumT / n as NumT).for_each(|x| hist.insert(x));

        let elem_per_bucket = n / nbuckets;
        for i in 0..nbuckets {
            for j in 0..elem_per_bucket {
                let r = i * elem_per_bucket + j;
                let x = hist.get_bucket_with_rank(r);
                let q = hist.quantile_in_bucket(x.0, x.1, r);

                assert!((q - ((j+1) as NumT / elem_per_bucket as NumT)).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn approx_quantile() {
        let n = 200;
        let nbuckets = 10;
        let mut hist = Histogram::<()>::new(nbuckets);
        hist.set_min_max(0.0, 1.0);
        (0..n).map(|i| i as NumT / n as NumT).for_each(|x| hist.insert(x));

        let cumsum = hist.cumsum();

        for i in 0..n {
            let q = i as NumT / n as NumT;
            let v = cumsum.approx_quantile(q);
            println!("{} -> {}", q, v);
            println!();
            assert!((q - v).abs() < 1e-2);
        }
    }
}
