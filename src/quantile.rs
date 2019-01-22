use std::borrow::{Borrow, BorrowMut};

use crate::NumT;


// Paper:
// Zhang, Qi, and Wei Wang. "A fast algorithm for approximate quantiles in high speed data
// streams." Scientific and Statistical Database Management, 2007. SSBDM'07. 19th International
// Conference on. IEEE, 2007.
//
// Also see: https://github.com/jasonge27/fastQuantile (MIT license)

//struct SummaryElem {
//    rmin: u32,
//    rmax: u32,
//    elem: NumT,
//}
//
//struct Summary<B>
//where B: Borrow<[SummaryElem]> {
//    elems: B,
//}
//
//impl <B> Summary<B>
//where B: Borrow<[SummaryElem]> {
//    pub fn quantile(&self, q: NumT) -> SummaryElem {
//        unimplemented!();
//    }
//
//}
//
//pub struct FixedLengthSummary {
//    eps: NumT,
//    blocksz: usize,
//    nlevels: usize,
//    levels: Vec<SummaryElem>,
//    len: usize,
//}
//
//impl FixedLengthSummary {
//    pub fn new(n: usize, eps: NumT) -> FixedLengthSummary {
//        let nf = n as NumT;
//        let blocksz = ((eps * nf).log2() / eps).floor() as usize;
//        let nlevels = (nf * eps).log2().ceil() as usize;
//        unimplemented!()
//    }
//
//    fn get_level(&self, i: usize) -> &[SummaryElem] {
//        assert!(i < self.nlevels);
//        let lo = i * self.blocksz;
//        let hi = (i+1) * self.blocksz;
//        &self.levels[lo..hi]
//    }
//
//    fn get_level_mut(&mut self, i: usize) -> &mut [SummaryElem] {
//        assert!(i < self.nlevels);
//        let lo = i * self.blocksz;
//        let hi = (i+1) * self.blocksz;
//        &mut self.levels[lo..hi]
//    }
//}




// ------------------------------------------------------------------------------------------------

use std::mem::transmute;

// f32 = [ 1 sign | 8 exp | 23 mantissa = 11 + 12 ]

#[derive(Debug, Clone)]
pub struct ApproxQuantileStats {
    neg_exp_min: u8,
    neg_exp_max: u8, // nb possible values (max_neg_exp-min_neg_exp+1)
    pos_exp_min: u8,
    pos_exp_max: u8,
}

impl ApproxQuantileStats {
    /// Take an empty float range, update by feeding actual values.
    pub fn new() -> ApproxQuantileStats {
        ApproxQuantileStats {
            neg_exp_min: 0xFF,
            neg_exp_max: 0x0,
            pos_exp_min: 0xFF,
            pos_exp_max: 0x0,
        }
    }

    /// Take the full float range; no need to feed.
    pub fn full_range() -> ApproxQuantileStats {
        ApproxQuantileStats {
            neg_exp_min: 0x0,  // -0.0
            neg_exp_max: 0xFF, // -Inf
            pos_exp_min: 0x0,  // +0.0
            pos_exp_max: 0xFF, // +Inf
        }
    }

    pub fn feed(&mut self, value: f32) {
        let bits = unsafe { transmute::<f32, u32>(value) };
        let exp = ((bits >> 23) & 0xFF) as u8;
        if bits & 0x80000000 == 0x80000000 { // negative values
            if bits != 0x80000000 { // not neg. zero
                self.neg_exp_min = self.neg_exp_min.min(exp);
                self.neg_exp_max = self.neg_exp_max.max(exp);
            }
        } else { // positive values
            if bits != 0x00000000 { // not pos. zero
                self.pos_exp_min = self.pos_exp_min.min(exp);
                self.pos_exp_max = self.pos_exp_max.max(exp);
            }
        }
    }
}

pub struct ApproxQuantile {
    count: u32,
    neg_exp_offset: u8,
    neg_exp_range: u8,
    pos_exp_offset: u8,
    neg_nbits_man: u8,
    pos_nbits_man: u8,
    level1: Vec<u32>, // max 4098 buckets, fits in L1 (12 bits of info)
    level2: Vec<u32>,
    buffer: Vec<f32>,
}

impl ApproxQuantile {
    pub fn new() -> ApproxQuantile {
        ApproxQuantile {
            count: 0,
            neg_exp_offset: 0x0,
            neg_exp_range: 0xFF,
            pos_exp_offset: 0x0, 
            neg_nbits_man: 3,
            pos_nbits_man: 3,
            level1: vec![0; 4096], // both should fit in L1 cache
            level2: vec![0; 2048],
            buffer: Vec::new(),
        }
    }

    pub fn reset(&mut self) {
        self.count = 0;
        self.level1.iter_mut().for_each(|x| *x = 0);
        self.level2.iter_mut().for_each(|x| *x = 0);
        self.buffer.clear();
    }

    pub fn set_stats(&mut self, stats: &ApproxQuantileStats) {
        if stats.neg_exp_min <= stats.neg_exp_max { // else, no neg values seen
            let d = stats.neg_exp_max as u32 - stats.neg_exp_min as u32 + 1;
            self.neg_nbits_man = 11 - (d as f32).log2().ceil() as u8; // 1 sign bit (12 total)!
            self.neg_exp_offset = stats.neg_exp_min;
            self.neg_exp_range = (d - 1) as u8;
        }

        if stats.pos_exp_min <= stats.pos_exp_max { // else, no pos values seen
            let d = stats.pos_exp_max as u32 - stats.pos_exp_min as u32 + 1;
            self.pos_nbits_man = 11 - (d as f32).log2().ceil() as u8;
            self.pos_exp_offset = stats.pos_exp_min;
        }
    }

    /// Get the mantissa bits we also include in level1.
    fn level1_sub_index(&self, bits: u32, nbits_man: u8) -> usize {
        let x = (bits & 0x007fffff) >> (23 - nbits_man as u32);
        x as usize
    }

    fn level1_bucket(&self, value: f32) -> usize {
        let bits = unsafe { transmute::<f32, u32>(value) };
        let sign = bits >> 31;
        let exp = ((bits >> 23) & 0xFF) as usize;

        let index = if sign == 1 { // neg. value
            let exp_index = self.neg_exp_range as usize - (exp - self.neg_exp_offset as usize);
            let bucket_base = (exp_index + 1) << self.neg_nbits_man;
            let sub_index = self.level1_sub_index(bits, self.neg_nbits_man);
            bucket_base - sub_index - 1
        } else { // pos. value
            let exp_index = exp - self.pos_exp_offset as usize;
            let bucket_base = exp_index << self.pos_nbits_man;
            let sub_index = self.level1_sub_index(bits, self.pos_nbits_man);
            0x800 + bucket_base + sub_index
        };

        index
    }

    pub fn feed(&mut self, value: f32) {
        let bucket = self.level1_bucket(value);
        //println!("index={}, bucket={} for {:.2e}", bucket >> 3, bucket, value);
        self.level1[bucket] += 1;
        self.count += 1;
    }
}






//pub struct ApproxQuantile {
//    count: u32,
//    level1: Vec<u32>, // first 9 bits -> 512 buckets (sign + exponent bits)
//    level2: Vec<u32>, // next 11 bits -> 2048 buckets
//    buffer: Vec<f32>,
//}
//
//impl ApproxQuantile {
//    pub fn new() -> ApproxQuantile {
//        ApproxQuantile {
//            count: 0,
//            level1: vec![0; 512],
//            level2: vec![0; 2048],
//            buffer: Vec::new(),
//        }
//    }
//
//    pub fn reset(&mut self) {
//        self.count = 0;
//        self.level1.iter_mut().for_each(|c| *c = 0);
//        self.level2.iter_mut().for_each(|c| *c = 0);
//        self.buffer.clear();
//    }
//
//    fn level1_bucket(value: f32) -> u32 {
//        let bits = unsafe { transmute::<f32, u32>(value) >> 23 }; // first 9 bits
//        if bits & 0x100 == 0x100 { 512 - bits } // sign bit is 1
//        else                     { bits + 256 }
//    }
//
//    fn level1_prefix(bucket: u32) -> u32 {
//        debug_assert!(0 < bucket && bucket < 512);
//        if bucket < 256 { 512 - bucket }
//        else            { bucket - 256 }
//    }
//
//    pub fn feed(&mut self, value: f32) {
//        debug_assert!(!value.is_nan());
//        debug_assert!(!value.is_infinite());
//        let bucket = Self::level1_bucket(value) as usize;
//        debug_assert!(bucket < self.level1.len());
//        unsafe { *self.level1.get_unchecked_mut(bucket) += 1 };
//        self.count += 1;
//    }
//
//    pub fn round2(&mut self, quantile: f32, eps: f32) -> ApproxQuantileRound2 {
//        debug_assert!(0.0 <= quantile && quantile <= 1.0);
//        let rank = 1 + (quantile * (self.count-1) as NumT).floor() as usize;
//        let (bucket, sub_rank) = get_bucket_containing_rank(&self.level1, rank);
//
//        dbg!(bucket);
//
//        ApproxQuantileRound2 {
//            eps,
//            sub_rank: sub_rank as u32,
//            prefix: Self::level1_prefix(bucket as u32),
//            level2: &mut self.level2,
//            buffer: &mut self.buffer,
//        }
//    }
//}
//
//pub struct ApproxQuantileRound2<'a> {
//    eps: f32,
//    sub_rank: u32,
//    prefix: u32,
//    level2: &'a mut [u32],
//    buffer: &'a mut Vec<f32>,
//}
//
//impl <'a> ApproxQuantileRound2<'a> {
//
//    fn level2_bucket(value: f32) -> u32 {
//        let bits = unsafe { transmute::<f32, u32>(value) };
//        (bits >> 12) & 0x7ff
//    }
//
//    pub fn feed(&mut self, value: f32) {
//        let bits = unsafe { transmute::<f32, u32>(value) };
//        //println!("{:09b} - {:09b} {}", bits >> 23, self.prefix, value);
//        if (bits >> 23) ^ self.prefix != 0 { return; } // only values that were selected in round1
//
//        let bucket = Self::level2_bucket(value) as usize;
//        debug_assert!(bucket < self.level2.len());
//        self.level2[bucket] += 1;
//        //unsafe { *self.level2.get_unchecked_mut(bucket) += 1 }
//        self.buffer.push(value);
//    }
//
//    pub fn get_quantile_value(&mut self) -> f32 {
//        for i in 0..2048 {
//            if self.level2[i] == 0 { continue; }
//            println!("{:4}: {}", i, self.level2[i]);
//        }
//
//        let (bucket, subsub_rank) = get_bucket_containing_rank(self.level2, self.sub_rank as usize);
//
//        unimplemented!()
//    }
//}

/// Returns index of bucket and rank inside bucket
fn get_bucket_containing_rank(buckets: &[u32], rank: usize) -> (usize, usize) {
    let mut accum = 0;
    let mut bucket = 0;
    dbg!(rank);
    loop {
        debug_assert!(bucket < buckets.len());
        let bucket_size = buckets[bucket] as usize;
        accum += bucket_size;
        if accum >= rank { break; }
        bucket += 1;
    }
    (bucket, accum - rank)
}



#[cfg(test)]
mod test {
    use super::{ApproxQuantileStats, ApproxQuantile};
    use std::mem::transmute;

    #[test]
    fn stats() {
        let n = 100000;
        let v: Vec<f32> = (0..n).map(|i| (i as f32 / n as f32) - (n/2) as f32 / (n-100) as f32).collect();
        let mut s = ApproxQuantileStats::new();

        v.iter().for_each(|&x| s.feed(x));

        dbg!(&s);
        panic!();
    }

    //fn level1_bits() {
    //    // negative numbers
    //    for i in 1u32..256u32 {
    //        let bits = 0b1_0000_0000 | (256-i);
    //        let f = unsafe { std::mem::transmute::<u32, f32>(bits << 23) };
    //        let bucket = ApproxQuantile::level1_bucket(f);
    //        let prefix = ApproxQuantile::level1_prefix(i);
    //        println!("-{:3}: {:09b} {:.5e}, bucket={}, prefix={:09b}", i, bits, f, bucket, prefix);
    //        assert_eq!(i, bucket);
    //        assert_eq!(bits, prefix);
    //    }

    //    // positive numbers
    //    for i in 0u32..256u32 {
    //        let f = unsafe { std::mem::transmute::<u32, f32>(i << 23) };
    //        let bucket = ApproxQuantile::level1_bucket(f);
    //        let prefix = ApproxQuantile::level1_prefix(i+256);
    //        println!("+{:3}: {:09b} {:.5e}, bucket={}, prefix={:09b}", i, i, f, bucket, prefix);
    //        assert_eq!(i+256, bucket);
    //    }
    //}

    #[test]
    fn basic_bucket0() {
        let n = 512;
        let f = |i| {
            let bits = if i < 256 { ((255-i) as u32 ^ 0b1_0000_0000) << 23 }
                       else       { (     i  as u32 ^ 0b1_0000_0000) << 23 };
            unsafe { transmute::<u32, f32>(bits) }
        };
        let v: Vec<f32> = (0..n).map(f).collect();

        let mut q = ApproxQuantile::new();
        v.iter().for_each(|&x| q.feed(x));

        for i in 0..n {
            let j = if i < 256 { ((i+1) << 3) - 1 } else { i << 3 };
            println!("j={} for i={}", j, i);
            let count = q.level1[j];
            assert_eq!(count, 1);
        }
    }

    #[test]
    fn basic_bucket1_full_range() {
        let n = 4096;
        let f = |i| {
            let bits = (((i>>3) as u32) ^ 0b1_0000_0000) << 23;
            let bits = bits | ((i as u32 & 0b111) << 20);
            unsafe { transmute::<u32, f32>(bits) }
        };
        let v: Vec<f32> = (0..n).map(f).collect();

        let mut q = ApproxQuantile::new();
        v.iter().for_each(|&x| q.feed(x));

        for i in 0..n {
            let count = q.level1[i];
            assert_eq!(count, 1);
        }
    }

    #[test]
    fn basic_bucket2_even_non_full() {
        let n = 4096;
        let f = |i| {
            let i = i as u32;
            let sign = ((i>>3) & 0b1_0000_0000) ^ 0b1_0000_0000;
            let exp =  ((i>>3) & 0b0_1110_0000) >> 5;
            let man =  (i & 0b0_1111_1111) << 15;
            let bits = ((sign | (exp + 127)) << 23) | man;
            unsafe { transmute::<u32, f32>(bits) }
        };
        let v: Vec<f32> = (0..n).map(f).collect();

        let mut s = ApproxQuantileStats::new();
        v.iter().for_each(|&x| s.feed(x));

        dbg!(&s);

        let mut q = ApproxQuantile::new();
        q.set_stats(&s);

        dbg!(q.neg_exp_offset);
        dbg!(q.neg_exp_range);
        dbg!(q.neg_nbits_man);
        dbg!(q.pos_nbits_man);

        v.iter().for_each(|&x| q.feed(x));

        for i in 0..n {
            let count = q.level1[i];
            assert_eq!(count, 1);
        }
    }
    
    #[test]
    fn basic_bucket3_uneven() {
        let n = 4096;
        let fneg = |i| {
            let bits = (((i>>3) as u32) ^ 0b1_0000_0000) << 23;
            let bits = bits | ((i as u32 & 0b111) << 20);
            unsafe { transmute::<u32, f32>(bits) }
        };
        let fpos = |i| {
            let i = i as u32;
            let exp =  ((i>>3) & 0b0_1110_0000) >> 5;
            let man =  (i & 0b0_1111_1111) << 15;
            let bits = ((exp + 127) << 23) | man;
            unsafe { transmute::<u32, f32>(bits) }
        };
        let mut v1: Vec<f32> = (0..2048).map(fneg).collect();
        let mut v2: Vec<f32> = (0..2048).map(fpos).collect();

        let mut v = Vec::new();
        v.append(&mut v1);
        v.append(&mut v2);

        {
            let mut s = ApproxQuantileStats::new();
            let mut q = ApproxQuantile::new();
            v.iter().for_each(|&x| s.feed(x));
            q.set_stats(&s);
            v.iter().for_each(|&x| q.feed(x));

            for i in 0..n {
                let count = q.level1[i];
                assert_eq!(count, 1);
            }
        }

        v.iter_mut().for_each(|x| *x = -*x);

        {
            let mut s = ApproxQuantileStats::new();
            let mut q = ApproxQuantile::new();
            v.iter().for_each(|&x| s.feed(x));
            q.set_stats(&s);
            v.iter().for_each(|&x| q.feed(x));

            for i in 0..n {
                let count = q.level1[i];
                assert_eq!(count, 1);
            }
        }
    }
}
