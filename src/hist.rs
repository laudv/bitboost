use crate::NumT;

pub struct Histogram<D>
where D: Default {
    min_value: NumT,
    max_value: NumT,
    delta: NumT,
    count: u32,

    buckets: Vec<(u32, D)>, // count, data
    cumsum: Vec<(u32, D)>,
}

impl <D> Histogram<D>
where D: Default {
    pub fn new(nbuckets: usize) -> Histogram<D> {
        let buckets = (0..nbuckets).map(|_| Default::default()).collect();
        let cumsum = (0..nbuckets+1).map(|_| Default::default()).collect();
        Histogram {
            min_value: 0.0,
            max_value: 0.0,
            delta: 0.0,
            count: 0,
            buckets,
            cumsum,
        }
    }

    pub fn count(&self) -> u32 { self.count }

    pub fn set_min_max(&mut self, min_value: NumT, max_value: NumT) {
        debug_assert!(min_value.is_finite());
        debug_assert!(max_value.is_finite());
        debug_assert!(min_value < max_value);
        self.min_value = min_value;
        self.max_value = max_value;
        self.delta = (max_value - min_value) / self.buckets.len() as NumT;
    }

    fn get_bucket(&self, value: NumT) -> usize {
        debug_assert!(self.min_value != self.max_value);
        debug_assert!(self.delta > 0.0);

        let x = (value - self.min_value) / self.delta;
        let i = x.floor() as isize;
        let i = (i.max(0) as usize).min(self.buckets.len() - 1);

        i
    }

    fn get_bucket_representative(&self, bucket: usize) -> NumT {
        self.min_value + bucket as NumT * self.delta
    }

    /// For repeated rank queries, use `cumsum`.
    fn get_bucket_with_rank(&self, rank: usize) -> (usize, u32) {
        debug_assert!(rank < u32::max_value() as usize);
        let rank = rank as u32;
        debug_assert!(rank < self.count);

        let mut count_accum = 0;
        let mut bucket = 0;
        for b in &self.buckets {
            let count = b.0;
            bucket += 1;
            if count_accum + count > rank { break; }
            count_accum += count;
        }

        let bucket = bucket - 1;

        debug_assert!(bucket < self.buckets.len());

        (bucket, count_accum) // bucket index, accumulated element count
    }

    fn quantile_in_bucket(&self, bucket: usize, cumsum: u32, rank: usize) -> NumT {
        let bucket_count = self.buckets[bucket].0;
        debug_assert_ne!(bucket_count, 0);
        debug_assert!(rank >= cumsum as usize);
        let diff = (1 + rank - cumsum as usize) as NumT;
        diff / bucket_count as NumT
    }

    fn rank_from_quantile(&self, quantile: NumT) -> usize {
        (quantile * (self.count - 1) as NumT).round() as usize
    }

    fn approx_quantile_interpol_aux(&self, bucket: usize, cumsum: u32, rank: usize) -> NumT {
        let u = self.quantile_in_bucket(bucket, cumsum, rank);
        let v1 = self.get_bucket_representative(bucket);
        let v2 = self.get_bucket_representative(bucket + 1);
        //println!("bucket: {} -> u={}, v1={}, v2={} => {}", bucket, u, v1, v2, (1.0 - u) * v1 + u * v2);
        (1.0 - u) * v1 + u * v2
    }

    pub fn approx_quantile_interpol(&self, quantile: NumT) -> NumT {
        let rank = self.rank_from_quantile(quantile);
        let (bucket, cumsum) = self.get_bucket_with_rank(rank);
        self.approx_quantile_interpol_aux(bucket, cumsum, rank)
    }

    pub fn approx_quantile(&self, quantile: NumT) -> (u32, NumT) {
        let rank = self.rank_from_quantile(quantile);
        let (bucket, r0) = self.get_bucket_with_rank(rank);
        let rank = rank as u32;
        let v0 = self.get_bucket_representative(bucket);
        if bucket < self.buckets.len() {
            let v1 = self.get_bucket_representative(bucket + 1);
            let r1 = r0 + self.buckets[bucket].0;
            if rank - r0 < r1 - rank { (r0, v0) }
            else                     { (r1, v1) }
        } else { (r0, v0) }
    }

    pub fn cumsum_with<'a, F>(&'a mut self, accumf: F) -> CumSum<'a, D>
    where D: Copy,
          F: Fn(D, D) -> D,
    {
        let mut count_accum = 0;
        let mut data_accum = Default::default();
        for i in 0..self.buckets.len() {
            count_accum += self.buckets[i].0;
            data_accum = accumf(data_accum, self.buckets[i].1);
            self.cumsum[i+1] = (count_accum, data_accum);
        }
        CumSum { hist: self }
    }


    pub fn reset(&mut self) {
        self.buckets.iter_mut().for_each(|x| *x = Default::default());
        self.count = 0;
    }
}

impl Histogram<()> {
    pub fn insert(&mut self, value: NumT) {
        let bucket = self.get_bucket(value);
        self.buckets[bucket].0 += 1;
        self.count += 1;
    }

    pub fn cumsum<'a>(&'a mut self) -> CumSum<'a, ()> {
        self.cumsum_with(|_, _| ())
    }
}

impl Histogram<NumT> {
    pub fn insert(&mut self, value: NumT, data: NumT) {
        let bucket = self.get_bucket(value);
        let b = &mut self.buckets[bucket];
        b.0 += 1;
        b.1 += data;
        self.count += 1;
    }

    pub fn cumsum<'a>(&'a mut self) -> CumSum<'a, NumT> {
        self.cumsum_with(|acc, d| acc + d)
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
        match self.hist.cumsum.binary_search_by_key(&rank, |p| p.0) {
            Ok(bucket)  => bucket,
            Err(bucket) => bucket-1,
        }
    }

    pub fn approx_quantile_interpol(&self, quantile: NumT) -> NumT {
        let rank = self.hist.rank_from_quantile(quantile);
        let bucket = self.get_bucket_with_rank(rank);
        let cumsum = self.hist.cumsum[bucket].0;
        println!("approx_quantile_interpol {} {} {}", bucket, cumsum, rank);
        self.hist.approx_quantile_interpol_aux(bucket, cumsum, rank)
    }

    /// Get number of cumulated examples, approximate quantile value and sum of data.
    pub fn approx_quantile_and_data(&self, quantile: NumT) -> (u32, NumT, D)
    where D: Copy {
        let rank = self.hist.rank_from_quantile(quantile);
        let bucket = self.get_bucket_with_rank(rank);
        let v0 = self.hist.get_bucket_representative(bucket);
        let (r0, d0) = self.hist.cumsum[bucket];

        if bucket < self.hist.buckets.len() {
            let v1 = self.hist.get_bucket_representative(bucket + 1);
            let (r1, d1) = self.hist.cumsum[bucket + 1];
            let df0 = rank as u32 - r0;
            let df1 = r1 - rank as u32;
            if df0 < df1 { (r0, v0, d0) }
            else         { (r1, v1, d1) }
        } else { (r0, v0, d0) }
    }

    pub fn approx_quantile(&self, quantile: NumT) -> (u32, NumT)
    where D: Copy {
        let (count, value, _) = self.approx_quantile_and_data(quantile);
        (count, value)
    }
}

impl <'a> CumSum<'a, ()> {
}

impl <'a> CumSum<'a, NumT> {
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
    fn approx_quantile_interpol() {
        let n = 200;
        let nbuckets = 10;
        let mut hist = Histogram::<()>::new(nbuckets);
        hist.set_min_max(0.0, 1.0);
        (0..n).map(|i| i as NumT / n as NumT).for_each(|x| hist.insert(x));

        let qs: Vec<NumT> = (0..n).map(|i| hist.approx_quantile_interpol(i as NumT / n as NumT)).collect();

        let cumsum = hist.cumsum();

        for i in 0..n {
            let q = i as NumT / n as NumT;
            let v1 = qs[i];
            dbg!(q);
            let v2 = cumsum.approx_quantile_interpol(q);
            println!("{} -> {}/{}", q, v1, v2);
            assert!((q - v2).abs() < 1e-2);
            assert!((v1 - v2).abs() < 1e-6);
        }
    }

    #[test]
    fn approx_quantile() {
        let n = 245;
        let nbuckets = 10;
        let mut hist = Histogram::<NumT>::new(nbuckets);
        hist.set_min_max(10.0, 20.0);
        (0..n).map(|i| 10.0 + 10.0 * (i as NumT / n as NumT)).for_each(|x| hist.insert(x, x));

        for i in 0..nbuckets+1 {
            let q = i as NumT / nbuckets as NumT;
            let (r0, v0) = hist.approx_quantile(q);
            let (r1, v1, d) = hist.cumsum().approx_quantile_and_data(q);
            let q0 = 10.0 + 10.0 * q;
            println!("hist:   {} -> {}, r={}", q, v0, r0);
            println!("cumsum: {} -> {}, r={}, data={}", q, v1, r1, d);
            assert!((q-q0).abs() < 1e6);
            assert_eq!(r0, r1);
            assert!((v0-v1).abs() < 1e6)
        }

        let (r, _, _) = hist.cumsum().approx_quantile_and_data(1.1);
        assert_eq!(r, n);
    }
}
