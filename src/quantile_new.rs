use std::ops::{Add};
use std::mem::{size_of};

use num::Num;


struct Bucketer<T, A>
where T: Num + Copy,
      A: Fn(T, T) -> T,
{
    offset: usize, // number of MSB that are ignored for bucketing
    nbits: usize,  // number of MSB after `offset` used for bucketing
    prefix: u32,   // the bits of the `offset` MSB bits (or zero if not used)
    buckets: Vec<(u32, T)>,

    accumulator: A,
}

impl <T, A> Bucketer<T, A>
where T: Num + Copy,
      A: Fn(T, T) -> T,
{
    fn new(offset: usize, accumulator: A) -> Bucketer<T, A> {
        let max_memory = 16*1024;
        let mut size = max_memory;
        let mut nbits = 14; // log2(max_memory)
        while size * size_of::<T>() > max_memory { size /= 2; nbits -= 1; }

        Bucketer {
            offset,
            nbits,
            prefix: 0,
            buckets: vec![(0, T::zero()); size],
            accumulator,
        }
    }

    fn set_offset(&mut self, offset: usize) { self.offset = offset; }
    fn set_prefix(&mut self, prefix: u32)   { self.prefix = prefix; }

    fn reset(&mut self) {
        self.buckets.iter_mut().for_each(|x| *x = (0, T::zero()));
    }

    fn get_bucket(&self, value: u32) -> usize {
        let mut bucket = value;
        bucket <<= self.offset;
        bucket >>= self.offset;
        bucket >>= 32 - self.offset - self.nbits;
        bucket as usize
    }

    fn get_bucket_representative(&self, bucket: usize) -> u32 {
        let mut value = bucket as u32;
        value <<= 32 - self.offset - self.nbits;
        value ^= self.prefix;
        value
    }

    fn feed(&mut self, value: u32, assoc: T) {
        let bucket = self.get_bucket(value);
        let bucket_value = &mut self.buckets[bucket];
        bucket_value.0 += 1;
        bucket_value.1 = (self.accumulator)(bucket_value.1, assoc);
    }

    fn get_bucket_containing_rank(&self, rank: usize) -> (usize, u32) {
        assert!(rank < u32::max_value() as usize);
        let rank = rank as u32;
        let mut count_accum = 0;
        let mut bucket = 0;
        for (count, _) in &self.buckets {
            let x = count_accum + count;
            if x > rank { break; }
            count_accum = x;
            bucket += 1;
        }
        (bucket, count_accum)
    }
}


pub struct FloatStats {
    min_exp: u8,
    max_exp: u8,
}


struct FloatBucketer<T, A>
where T: Num + Copy,
      A: Fn(T, T) -> T,
{
    bucketer: Bucketer<T, A>,
}

impl <T, A> FloatBucketer<T, A>
where T: Num + Copy,
      A: Fn(T, T) -> T,
{
    fn new(min_exp: u8, max_exp: u8, accumulator: A) -> FloatBucketer<T, A> {
        //let bucketer = Bucketer::new
        //FloatBucketer {
        //    bucketer,
        //}
        unimplemented!()
    }
}











#[cfg(test)]
mod test {
    use super::Bucketer;

    #[test]
    fn bucketer1() {
        let n = 1_000_000;
        let mut bucketer = Bucketer::new(1, |a: f32, b| a.max(b));

        for i in 0..n {
            let x = (i as f32 / n as f32).to_bits();
            bucketer.feed(x, f32::from_bits(x));
        }

        for (i, (b, x)) in bucketer.buckets.iter().enumerate() {
            if *b == 0 { continue; }
            println!("i={:4}: {}, {}", i, b, x);
        }

        panic!()
    }

}
