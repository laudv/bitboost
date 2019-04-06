/*
 * Copyright (c) DTAI - KU Leuven – All rights reserved.
 * Proprietary, do not copy or distribute without permission.
 * Written by Laurens Devos, 2019
*/

use std::ops::Range;
use std::slice::Iter;
use std::iter::Cloned;

use rand::{Rng, SeedableRng};
use rand::rngs::SmallRng;

use crate::{NumT, CatT, into_cat, EPSILON};
use crate::config::Config;
use crate::data::{Data, FeatType};
use crate::slice_store::{SliceRange, BitBlockStore, BitVecRef};
use crate::binner::Binner;
use crate::simd;

const QUANTILE_EST_NBINS: usize = 512;

pub struct InnerDataset<'a> {
    rng: SmallRng,
    data: &'a Data,
    useful_features: Vec<usize>, // == first `nactive_features` are 'active'
    nactive_features: usize,
    nactive_examples: usize,     // == nexamples in bag
    store: BitBlockStore,
    bitvecs: Vec<Vec<SliceRange>>,
    supercats: Vec<Vec<CatT>>,
    split_candidates: Vec<Vec<NumT>>,
    used_nbins: Vec<usize>,
    has_non_locard_cat_features: bool,
}

impl <'a> InnerDataset<'a> {
    fn new(config: &Config, data: &'a Data) -> Self {
        let rng = SmallRng::seed_from_u64(config.random_seed);
        let mut has_non_locard_cat_features = false;
        let mut useful_features = Vec::new();
        let mut store = BitBlockStore::new(2048);

        let nfeatures = data.nfeatures();
        let nexamples = data.nexamples();
        let nactive_examples = (nexamples as NumT * config.example_fraction).round() as usize;

        let mut bitvecs      = vec![Vec::new(); nfeatures];
        let split_candidates = vec![Vec::new(); nfeatures];
        let used_nbins       = vec![0; nfeatures];
        let supercats        = Vec::new();

        // find useful features and allocate bitvecs
        for feat_id in 0..data.nfeatures() {
            let max_nbins = data.max_nbins(feat_id);
            if max_nbins == 0 { continue; } // not a useful feature, only 1 value

            let feat_bounds = data.feat_limits(feat_id);
            if feat_bounds.0 == feat_bounds.1 { continue; } // not a useful feature, only 1 value

            has_non_locard_cat_features |= data.feat_type(feat_id) != FeatType::LoCardCat;

            useful_features.push(feat_id);
            bitvecs[feat_id] = (0..max_nbins)
                .map(|_| store.alloc_zero_bits(nactive_examples))
                .collect();
        }

        let nuseful = useful_features.len();
        let nactive_features = (nuseful as NumT * config.feature_fraction).round() as usize;

        Self {
            rng,
            data,
            useful_features,
            nactive_features,
            nactive_examples,
            store,
            bitvecs,
            supercats,
            split_candidates,
            used_nbins,
            has_non_locard_cat_features,
        }
    }

    fn feature_sampling_enabled(&self) -> bool {
        self.useful_features.len() > self.nactive_features
    }

    fn sample_features(&mut self) {
        if self.feature_sampling_enabled() {
            shuffle_first_k(&mut self.rng, self.nactive_features, &mut self.useful_features);
        }
    }

    fn active_features<'b>(&'b self) -> &'b [usize] {
        &self.useful_features[0..self.nactive_features]
    }

    fn initialize_supercats(&mut self) {
        self.supercats.resize(self.data.nfeatures(), Vec::new());
    }

    fn update_locard_cat<I>(&mut self, _config: &Config, feat_id: usize, example_iter: I)
    where I: IntoIterator<Item = usize> + Copy
    {
        let data = self.data.get_feature(feat_id);
        let get_cat = |i| into_cat(data[i]) as usize;
        self.used_nbins[feat_id] = self.data.max_nbins(feat_id);
        debug_assert_eq!(self.used_nbins[feat_id], self.bitvecs[feat_id].len());
        Self::zero_bitvecs(&mut self.store, &self.bitvecs[feat_id]);
        Self::fill_bitvecs(&mut self.store, &self.bitvecs[feat_id], example_iter, get_cat);
    }

    fn update_hicard_cat<I>(&mut self, config: &Config, feat_id: usize, example_iter: I,
                            grad: &[NumT], grad_bounds: (NumT, NumT))
    where I: IntoIterator<Item = usize> + Copy
    {
        let data = self.data.get_feature(feat_id);
        let card = self.data.feat_card(feat_id);

        // collect gradient sums & counts per categorical value
        let mut grad_stat_pairs: Vec<(NumT, u32)> = vec![(0.0, 0); card];
        for (i, j) in example_iter.into_iter().enumerate() {
            let category = into_cat(data[j]) as usize;
            let entry = &mut grad_stat_pairs[category];
            entry.0 += grad[i];
            entry.1 += 1;
        }

        // accumulate category values (mean) -> this determines their ordening
        // combine similar categories using quantile estimation into 'super-categories'
        let mut bins = vec![0u32; QUANTILE_EST_NBINS];
        let mut binner = Binner::new(&mut bins, grad_bounds);
        let combiner = |bin: &mut u32, count: u32| *bin += count;
        for (sum, count) in grad_stat_pairs.iter_mut() {
            if *count == 0 { continue; }
            *sum = *sum / *count as NumT; // = category weight
            binner.insert(*sum, *count, combiner);
        }

        // extract quantiles
        let extractor = |bin: &u32| *bin;
        let rank_step = self.nactive_examples as NumT / (config.max_nbins + 1) as NumT;
        let ranks = (1..=config.max_nbins).map(|i| (i as NumT * rank_step).round() as u32 - 1);
        let qbins = binner.rank_iter(ranks, extractor);
        let mut last_bin = usize::max_value();
        let mut split_weights = Vec::with_capacity(config.max_nbins);
        for bin in qbins {
            if bin == last_bin { continue; }
            last_bin = bin;
            split_weights.push(binner.bin_representative(bin + 1)); // -1 ^ and +1?
        }
        let supercard = split_weights.len();
        debug_assert!(supercard <= config.max_nbins);

//        println!("split_weights feature {} [#card {} #weight {}]: {:?}", feat_id, card,
//                 supercard, split_weights);

        // generate mapping: category -> super-category
        let mut supercats = vec![0; card];
        for (category, &(mean, _)) in grad_stat_pairs.iter().enumerate() {
            //let supercat = split_weights.binary_search_by(|&x| {
            //    if x < mean { Ordering::Less } else { Ordering::Greater }
            //}).expect_err("nothing is equal, see cmp");
            let mut supercat = 0;
            for &split in &split_weights {          // lin.search is 15-20% faster than bin.search
                if split < mean { supercat += 1; }  // split_weights.len() is small
                else { break; }
            }
            supercats[category] = supercat as CatT;
        }

        // generate bitvecs
        let bitvecs = &self.bitvecs[feat_id][0..supercard];
        let get_cat = |i| supercats[into_cat(data[i]) as usize] as usize;
        Self::zero_bitvecs(&mut self.store, bitvecs);
        Self::fill_bitvecs(&mut self.store, bitvecs, example_iter, get_cat);
        Self::accumulate_bitvecs(&mut self.store, bitvecs);

        self.used_nbins[feat_id] = supercard;
        self.supercats[feat_id] = supercats;
    }

    fn update_num<'b, I>(&mut self, config: &Config, feat_id: usize, example_iter: I,
                         grad: &[NumT])
    where I: IntoIterator<Item = usize> + Copy
    {
        let data = self.data.get_feature(feat_id);
        let feat_bounds = self.data.feat_limits(feat_id);

        // quantile est. weighted by grad. vals so there is a variation in the splits we generate
        let mut bins: Vec<NumT> = vec![0.0; QUANTILE_EST_NBINS];
        let mut binner = Binner::new(&mut bins, feat_bounds);
        let combiner = |bin: &mut NumT, grad_value: NumT| *bin += grad_value;
        let mut grad_weight_sum = 0.0;
        for j in example_iter.into_iter() { // XXX apply transformation to grad weights?
            let (feat_value, grad_value) = (data[j], grad[j].abs() + EPSILON);
            grad_weight_sum += grad_value;
            binner.insert(feat_value, grad_value, combiner);
        }

        // extract approximate quantiles
        let weight_step = grad_weight_sum / (config.max_nbins + 1) as NumT;
        let weights = (1..=config.max_nbins).map(|i| i as NumT * weight_step);
        let qbins = binner.rank_iter(weights, |bin| *bin);
        let mut last_bin = usize::max_value();
        let mut split_candidates = Vec::with_capacity(config.max_nbins);
        for bin in qbins {
            if bin == last_bin { continue; }
            last_bin = bin;
            split_candidates.push(binner.bin_representative(bin + 1));
        }
        let nsplit_candidates = split_candidates.len();
        assert!(nsplit_candidates <= config.max_nbins);
        assert!(nsplit_candidates > 0);

        // we don't want to have to search through the split candidates for each example to map the
        // examples to the split categories, so we look at the the split categories of the bins we
        // used earlier, and then map all the examples to the bins again
        let mut bins: Vec<CatT> = unsafe {  // reuse memory for `bins`, avoiding reallocation
            debug_assert_eq!(std::mem::size_of::<CatT>(), std::mem::size_of::<NumT>());
            let bins_ptr = bins.as_mut_ptr() as *mut CatT;
            let bins_cap = bins.capacity();
            std::mem::forget(bins);
            Vec::from_raw_parts(bins_ptr, QUANTILE_EST_NBINS, bins_cap)
        };
        let mut binner = Binner::new(&mut bins, feat_bounds);
        for i in 0..QUANTILE_EST_NBINS {
            let rep = binner.bin_representative(i + 1);
            let mut cat = 0;
            for &split in &split_candidates { // linear search is 15-20% faster than binary search
                if split < rep { cat += 1; }
                else { break; }
            }
            binner.bins_mut()[i] = cat as CatT;
        }

        // construct bitvecs
        let bitvecs = &self.bitvecs[feat_id][0..nsplit_candidates];
        let get_cat = |i| {
            let feat_value = data[i];
            //let cat = split_candidates.binary_search_by(|&split_cand| {
            //    if split_cand < feat_value { Ordering::Less }
            //    else                       { Ordering::Greater }
            //}).expect_err("nothing is equal, see cmp");
            //let mut cat = 0;
            //for &split in &split_candidates { // linear search is 15-20% faster than binary search
            //    if split < feat_value { cat += 1; }
            //    else { break; }
            //}
            //println!("NUM CAT = {} [ max={} feat_value={} < {:?}]", cat, nsplit_candidates,
            //         feat_value, split_candidates.get(cat));
            let alt_cat = binner.bin_value(binner.get_bin(feat_value));
            //println!("alt_cat = {}, cat = {}", alt_cat, cat);
            //cat
            alt_cat as usize
        };
        Self::zero_bitvecs(&mut self.store, bitvecs);
        Self::fill_bitvecs(&mut self.store, bitvecs, example_iter, get_cat);
        Self::accumulate_bitvecs(&mut self.store, bitvecs);

        self.used_nbins[feat_id] = nsplit_candidates;
        self.split_candidates[feat_id] = split_candidates;
    }

    fn zero_bitvecs(store: &mut BitBlockStore, bitvecs: &[SliceRange]) {
        for &range in bitvecs {
            store.get_bitvec_mut(range).cast_mut::<u64>().iter_mut().for_each(|x| *x = 0);
        }
    }

    fn fill_bitvecs<I, F>(store: &mut BitBlockStore, bitvecs: &[SliceRange], example_iter: I,
                          get_cat: F)
    where I: IntoIterator<Item = usize>,
          F: Fn(usize) -> usize
    {
        let max_category = bitvecs.len();
        for (i, j) in example_iter.into_iter().enumerate() {
            let category = get_cat(j);               // category == max_category if value belongs..
            debug_assert!(category <= max_category); // ..to last implicit category
            if category < max_category {
                store.get_bitvec_mut(bitvecs[get_cat(j)]).enable_bit(i);
            }
        }
    }

    fn accumulate_bitvecs(store: &mut BitBlockStore, bitvecs: &[SliceRange]) {
        for (&r0, &r1) in bitvecs[0..].iter().zip(bitvecs[1..].iter()) {
            let (bv0, mut bv1) = store.get_two_bitvecs_mut(r0, r1);
            unsafe { simd::or_assign(&mut bv1, &bv0); }
        }
    }
}









// ------------------------------------------------------------------------------------------------

pub struct Dataset<'a> {
    inner: InnerDataset<'a>,
    index_buffer: Vec<usize>, // 0..k examples are active examples, empty if ex.sampling disabled
    update_count: usize,
}

impl <'a> Dataset<'a> {
    pub fn new(config: &Config, data: &'a Data) -> Self {
        let mut inner = InnerDataset::new(config, data);
        let mut index_buffer = Vec::new();
            
        if config.example_fraction < 1.0 {
            // bagging is enabled, choose examples
            index_buffer = (0..inner.data.nexamples()).collect();
        } else {
            // no bagging, update locard-cat once and reuse throughout boosting
            let all_examples = RangeIntoIter(inner.data.nexamples());
            for u in 0..inner.useful_features.len() {
                let feat_id = inner.useful_features[u];
                if inner.data.feat_type(feat_id) == FeatType::LoCardCat {
                    inner.update_locard_cat(config, feat_id, &all_examples);
                }
            }
        }

        Dataset {
            inner,
            index_buffer,
            update_count: 0,
        }
    }

    fn sample_examples(&mut self) {
        let k = self.nactive_examples();
        shuffle_first_k(&mut self.inner.rng, k, &mut self.index_buffer);
    }

    fn active_examples_noborrow<'b>(index_buffer: &'b [usize], nactive_examples: usize) -> &'b [usize] {
        &index_buffer[0..nactive_examples] // does not borrow `self`
    }

    fn active_examples<'b>(&'b self) -> &'b [usize] { // borrows `self`
        Self::active_examples_noborrow(&self.index_buffer, self.nactive_examples())
    }

    pub fn update(&mut self, config: &Config, grad: &[NumT], grad_bounds: (NumT, NumT)) {
        self.update_count += 1;
        self.inner.initialize_supercats();

        // only fully update every `config.sample_freq` updates
        if config.sample_freq == 0 || (self.update_count > 1 &&
                                       self.update_count % config.sample_freq != 0) { return; }

        self.inner.sample_features();

        // Example sampling
        if self.example_sampling_enabled() {
            self.sample_examples();

            let (buf, nactive) = (&self.index_buffer, self.nactive_examples());
            let active_examples = Self::active_examples_noborrow(buf, nactive);
            let examples = SliceIntoIter(active_examples);

            for u in 0..self.nactive_features() {
                let feat_id = self.inner.useful_features[u]; // [0..nactive_features] are active
                match self.inner.data.feat_type(feat_id) {
                    FeatType::LoCardCat => {
                        self.inner.update_locard_cat(config, feat_id, &examples);
                    },
                    FeatType::HiCardCat => {
                        self.inner.update_hicard_cat(config, feat_id, &examples, grad,
                                                     grad_bounds);
                    },
                    FeatType::Numerical => {
                        self.inner.update_num(config, feat_id, &examples, grad);
                    },
                }
            }
        }
        
        // No example sampling -- only if there are non-locard categorical features
        else if self.inner.has_non_locard_cat_features {
            let examples = RangeIntoIter(self.inner.data.nexamples());
            for u in 0..self.nactive_features() {
                let feat_id = self.inner.useful_features[u]; // [0..nactive_features] are active

                match self.inner.data.feat_type(feat_id) {
                    FeatType::LoCardCat => {}, // can reuse bitvecs from initialization
                    FeatType::HiCardCat => {
                        self.inner.update_hicard_cat(config, feat_id, &examples, grad,
                                                     grad_bounds);
                    },
                    FeatType::Numerical => {
                        self.inner.update_num(config, feat_id, &examples, grad)
                    },
                }
            }
        }
    }

    // ------------

    pub fn feature_sampling_enabled(&self) -> bool { self.inner.feature_sampling_enabled() }
    pub fn example_sampling_enabled(&self) -> bool { !self.index_buffer.is_empty() }
    pub fn nactive_features(&self) -> usize { self.inner.nactive_features }
    pub fn nactive_examples(&self) -> usize { self.inner.nactive_examples }
    pub fn active_features(&self) -> &[usize] { &self.inner.active_features() }

    pub fn map_index(&self, local_index: usize) -> usize {
        debug_assert!(self.example_sampling_enabled()); // bagged index -> global data index
        self.active_examples()[local_index]
    }

    pub fn active_examples_iter<F>(&self, mut body: F)
    where F: FnMut(usize, usize) { // local index, global index
        if self.example_sampling_enabled() {
            for (i, &j) in self.active_examples().iter().enumerate() { (body)(i, j); }
        } else {
            for i in 0..self.nactive_examples() { (body)(i, i); }
        }
    }

    pub fn inactive_examples_iter<'b>(&'b self) -> impl Iterator<Item=usize> + 'b {
        assert!(self.example_sampling_enabled());
        InactiveExampleIter {
            active_examples: &self.active_examples(),
            index: 0,
            n: self.inner.data.nexamples(),
        }
    }

    pub fn get_nbins(&self, feat_id: usize) -> usize {
        self.inner.used_nbins[feat_id]
    }

    pub fn get_bitvec(&self, feat_id: usize, split_id: usize) -> BitVecRef {
        let range = self.inner.bitvecs[feat_id][split_id];
        self.inner.store.get_bitvec(range)
    }

    pub fn get_split_value(&self, feat_id: usize, split_id: usize) -> NumT {
        match self.inner.data.feat_type(feat_id) {
            FeatType::LoCardCat => split_id as NumT,
            FeatType::HiCardCat => split_id as NumT, // == super-category
            FeatType::Numerical => self.inner.split_candidates[feat_id][split_id],
        }
    }

    pub fn get_supercats(&mut self) -> &Vec<Vec<CatT>> {
        //assert!(!self.inner.supercats.is_empty(), "not yet updated?");
        //let mut tmp = Vec::new();
        //std::mem::swap(&mut tmp, &mut self.inner.supercats);
        //debug_assert!(!tmp.is_empty());
        //debug_assert!(self.inner.supercats.is_empty());
        //tmp
        &self.inner.supercats
    }
    
    pub fn data(&self) -> &Data {
        self.inner.data
    }
}








// ------------------------------------------------------------------------------------------------

struct RangeIntoIter(usize);
struct SliceIntoIter<'a>(&'a [usize]);

impl <'a> IntoIterator for &'a RangeIntoIter {
    type Item = usize;
    type IntoIter = Range<usize>;
    fn into_iter(self) -> Self::IntoIter { 0..self.0 }
}

impl <'a> IntoIterator for &'a SliceIntoIter<'a> {
    type Item = usize;
    type IntoIter = Cloned<Iter<'a, usize>>;
    fn into_iter(self) -> Self::IntoIter { self.0.iter().cloned() }
}

pub struct InactiveExampleIter<'b> {
    active_examples: &'b [usize],
    index: usize,
    n: usize,
}

impl <'b> Iterator for InactiveExampleIter<'b> {
    type Item = usize;
    fn next(&mut self) -> Option<usize> {
        while self.active_examples.get(0).map_or(false, |&i| i == self.index) {
            self.active_examples = &self.active_examples[1..];
            self.index += 1;
        }
        if self.index < self.n {
            let tmp = self.index;
            self.index += 1;
            Some(tmp)
        } else { None }
    }
}

#[allow(dead_code)]
fn sample_replacement(rng: &mut SmallRng, n: usize, buffer: &mut [usize]) {
    buffer.iter_mut().for_each(|i| *i = rng.gen_range(0, n));
    buffer.sort_unstable();
}

fn shuffle_first_k(rng: &mut SmallRng, k: usize, buffer: &mut [usize]) {
    let n = buffer.len();
    for i in 0..k {
        let j = rng.gen_range(i, n);
        buffer.swap(i, j);
    }
    // buffer[0..k] is random sample w/o replacement from 0..n
    buffer[0..k].sort_unstable();
}

#[allow(dead_code)]
fn reservoir_sample(rng: &mut SmallRng, n: usize, buffer: &mut [usize]) {
    let k = buffer.len();
    debug_assert!(n > k);
    for i in 0..n {
        if i < k { buffer[i] = i; }
        else {
            let r = rng.gen_range(0, i+1);
            if r < k { buffer[r] = i; }
        }
    }
    buffer.sort_unstable();
}









// ------------------------------------------------------------------------------------------------

#[cfg(test)]
mod test {
    use crate::data::Data;
    use super::*;

    #[test]
    fn locard_cat() {
        let mut config = Config::new();
        config.csv_has_header = false;
        config.random_seed = 4;
        config.feature_fraction = 0.67;
        config.example_fraction = 0.75;
        config.categorical_features = vec![1];
        config.max_nbins = 4;

        let d = "1,0,3,0\n4,1,6,0\n7,1,9,0\n10,1,12,0\n13,0,15,0\n16,2,18,0\n19,2,21,0\n22,0,24,0";
        let data = Data::from_csv(&config, d).unwrap();

        assert_eq!(data.nexamples(), 8);
        assert_eq!(data.nfeatures(), 3);
        assert_eq!(data.feat_card(1), 3);

        let target = data.get_target();
        let target_lims = data.feat_limits(data.target_id());
        let mut dataset = Dataset::new(&config, &data);
        dataset.update(&config, target, target_lims);

        assert_eq!(dataset.nactive_features(), 2);
        assert_eq!(dataset.nactive_examples(), 6);
        assert_eq!(dataset.active_features(), &[0, 1]); // dependent on seed
        assert_eq!(dataset.active_examples(), &[0, 1, 3, 5, 6, 7]); // depends on seed

        let ranges = &dataset.inner.bitvecs[1]; // locard-cat feature
        let values = vec![0b100001, 0b110, 0b11000];
        for i in 0..3 {
            let bitvec = dataset.inner.store.get_bitvec(ranges[i]);
            let x = bitvec.cast::<u32>()[0];
            println!("{:3}: {:032b}", i, x);
            assert_eq!(values[i], x);
        }
    }

    #[test]
    fn hicard_cat() {
        let mut config = Config::new();
        config.csv_has_header = false;
        config.categorical_features = vec![0];
        config.max_nbins = 8;
        let d = "1,1\n1,1\n2,1\n2,1\n3,2\n3,2\n4,2\n4,2\n5,3\n5,3\n6,3\n6,3\n7,4\n7,4\n8,4\n8,4\n\
                 9,5\n9,5\n10,5\n10,5\n11,6\n11,6\n12,6\n12,6\n13,7\n13,7\n14,7\n14,7\n15,8\n15,8\
                 \n16,8\n16,8";
        let data = Data::from_csv(&config, d).unwrap();
        let target = data.get_target();
        let target_lims = data.feat_limits(data.target_id());
        let mut dataset = Dataset::new(&config, &data);
        dataset.update(&config, target, target_lims);

        let ranges = &dataset.inner.bitvecs[0];
        let values = vec![0b00000000000000000000000000001111u32,
                          0b00000000000000000000000011111111,
                          0b00000000000000000000111111111111,
                          0b00000000000000001111111111111111,
                          0b00000000000011111111111111111111,
                          0b00000000111111111111111111111111,
                          0b00001111111111111111111111111111];
        for i in 0..dataset.get_nbins(0) {
            let bitvec = dataset.inner.store.get_bitvec(ranges[i]);
            let x = bitvec.cast::<u32>()[0];
            println!("{:3}: {:032b}", i, x);
            assert_eq!(values[i], x);
        }
    }

    #[test]
    fn dataset_nbins() {
        let mut config = Config::new();
        config.csv_has_header = false;
        config.categorical_features = vec![0];
        config.max_nbins = 8;
        let d = "8,1\n7,1\n1,0\n7,1\n3,0\n8,1\n6,1\n2,0\n5,1\n4,1\n2,0\n7,1\n3,0\n8,1\n6,1\n3,0\n\
                 7,1\n5,1\n5,1\n4,1\n2,0\n1,0\n6,1\n2,0\n6,1\n1,0\n4,1\n3,0\n4,1\n8,1\n1,0\n5,1";
        let data = Data::from_csv(&config, d).unwrap();
        let target = data.get_target();
        let target_lims = data.feat_limits(data.target_id());
        let mut dataset = Dataset::new(&config, &data);
        dataset.update(&config, target, target_lims);

        assert_eq!(8, data.max_nbins(0));
        assert_eq!(2, dataset.get_nbins(0)); // 2 classes: <4, >=4 -> targets 0, 1
    }

    fn dataset_num_aux(data_str: &str, values: &[u32]) {
        let mut config = Config::new();
        config.csv_has_header = false;
        config.max_nbins = 8;
        let data = Data::from_csv(&config, data_str).unwrap();
        let target = data.get_target();
        let target_lims = data.feat_limits(data.target_id());
        let mut dataset = Dataset::new(&config, &data);
        dataset.update(&config, target, target_lims);

        //dbg!(&data.features);

        let ranges = &dataset.inner.bitvecs[0];
        for i in 0..dataset.get_nbins(0) {
            let bitvec = dataset.inner.store.get_bitvec(ranges[i]);
            let x = bitvec.cast::<u32>()[0];
            println!("{:3}: {:032b}", i, x);
            assert_eq!(values[i], x);
        }
    }

    #[test]
    fn dataset_num1() {
        let values = vec![0b00000000000000000000000000001111u32,
                          0b00000000000000000000000011111111,
                          0b00000000000000000000011111111111,
                          0b00000000000000001111111111111111,
                          0b00000000000000111111111111111111,
                          0b00000000001111111111111111111111,
                          0b00000001111111111111111111111111,
                          0b00011111111111111111111111111111];
        let d = "0,1\n6,1\n11,1\n11,1\n13,1\n21,1\n24,1\n31,1\n36,1\n38,1\n42,1\n48,1\n60,1\n60,1\
                 \n61,1\n61,1\n64,1\n68,1\n75,1\n80,1\n81,1\n84,1\n85,1\n86,1\n89,1\n90,1\n91,1\n\
                 92,1\n92,1\n93,1\n96,1\n98,1";
        dataset_num_aux(d, &values);
    }

    #[test]
    fn dataset_num2() {
        let values = vec![0b00000000000000000000011111111111u32, // less weight -> coarser splits
                          0b00000000000000001111111111111111,
                          0b00000000000001111111111111111111,
                          0b00000000001111111111111111111111,
                          0b00000000111111111111111111111111,
                          0b00000111111111111111111111111111,
                          0b00011111111111111111111111111111,
                          0b01111111111111111111111111111111]; // more weight -> finer splits
        let d = "0,1\n6,2\n11,3\n11,4\n13,5\n21,7\n24,8\n31,9\n36,10\n38,11\n42,12\n48,13\n60,14\
                 \n60,15\n61,16\n61,18\n64,19\n68,20\n75,21\n80,22\n81,23\n84,24\n85,25\n86,26\n\
                 89,27\n90,28\n91,30\n92,31\n92,32\n93,33\n96,34\n98,35";
        dataset_num_aux(d, &values);
    }

    #[test]
    fn dataset_all() {
        let mut config = Config::new();
        config.categorical_features = vec![0, 1];
        config.csv_has_header = false;
        config.max_nbins = 8;
        let d = "6,16,1,0.01\n4,19,2,0.02\n5,6,3,0.02\n0,4,4,0.03\n6,5,5,0.03\n4,4,6,0.04\n1,15,7,0.08\n2,16,8,0.09\n6,8,9,0.09\n4,14,10,0.09\n2,2,11,0.1\n5,11,12,0.13\n4,1,13,0.14\n0,9,14,0.18\n0,18,15,0.22\n3,12,16,0.22\n1,18,17,0.24\n0,8,18,0.27\n6,17,19,0.28\n3,14,20,0.28\n0,12,21,0.3\n6,16,22,0.32\n5,1,23,0.35\n0,13,24,0.36\n6,17,25,0.37\n3,10,26,0.37\n2,3,27,0.38\n6,9,28,0.4\n1,18,29,0.44\n5,7,30,0.45\n2,4,31,0.45\n6,5,32,0.49\n0,14,33,0.49\n2,19,34,0.49\n1,20,35,0.5\n4,3,36,0.53\n3,9,37,0.54\n6,20,38,0.6\n2,12,39,0.61\n6,11,40,0.62\n2,6,41,0.63\n0,8,42,0.65\n3,19,43,0.68\n4,13,44,0.7\n4,15,45,0.71\n5,2,46,0.74\n5,10,47,0.74\n6,3,48,0.75\n6,7,49,0.76\n6,15,50,0.76\n3,11,51,0.77\n5,2,52,0.8\n6,1,53,0.82\n2,7,54,0.84\n1,4,55,0.86\n6,13,56,0.88\n3,5,57,0.89\n3,20,58,0.92\n5,6,59,0.92\n1,1,60,0.94\n4,2,61,0.96\n6,17,62,0.99\n1,3,63,0.99\n1,10,64,0.99";
        let data = Data::from_csv(&config, d).unwrap();
        let target = data.get_target();
        let target_lims = data.feat_limits(data.target_id());
        let mut dataset = Dataset::new(&config, &data);
        dataset.update(&config, target, target_lims);

        assert_eq!(data.max_nbins(0), 7);
        assert_eq!(data.max_nbins(1), 8);
        assert_eq!(data.max_nbins(2), 8);
        assert_eq!(dataset.get_nbins(0), 7);
        assert_eq!(dataset.get_nbins(1), 8);
        assert_eq!(dataset.get_nbins(2), 8);

        let values = vec![0b0000000000000000000000100000000100000000100100100110000000001000,
                          0b1100100001000000000000000000010000010000000000010000000001000000,
                          0b0000000000100000000000010100001001000100000000000000010010000000,
                          0b0000001100000100000001000001000000000010000010001000000000000000,
                          0b0001000000000000000110000000100000000000000000000001001000100010,
                          0b0000010000001000011000000000000000100000010000000000100000000100,
                          0b0010000010010011100000001010000010001001001001000000000100010001,
                          0b0, // skip
                          
                          0b0000000000000000000000000000000100010000001010010100001010000001,
                          0b0000000001000000000000100000000101010000001010110100001110101001,
                          0b0000000001000000000000100101000101011000001110111110001110101001,
                          0b0000000101000000000001100101001111011000001110111110001110111011,
                          0b0000010101000110000101111101001111011000001110111110101111111111,
                          0b0010110101010110000101111101001111011001011111111111101111111111,
                          0b0011110111011110001111111101001111011001111111111111111111111111,
                          0b0111111111011110101111111111111111011101111111111111111111111111,
                          
                          0b0000000000000000000000000000000000000000011111111111111111111111,
                          0b0000000000000000000000000000000011111111111111111111111111111111,
                          0b0000000000000000000000000111111111111111111111111111111111111111,
                          0b0000000000000000000011111111111111111111111111111111111111111111,
                          0b0000000000000001111111111111111111111111111111111111111111111111,
                          0b0000000000011111111111111111111111111111111111111111111111111111,
                          0b0000000111111111111111111111111111111111111111111111111111111111,
                          0b0001111111111111111111111111111111111111111111111111111111111111u64];

        for k in 0..3 {
            println!("== feature {}", k);
            let ranges = &dataset.inner.bitvecs[k];
            for i in 0..dataset.get_nbins(k) {
                let bitvec = dataset.inner.store.get_bitvec(ranges[i]);
                let x = bitvec.cast::<u64>()[0];
                println!("{:3}: {:064b}", i, x);
                assert_eq!(values[k * 8 + i], x);
            }
            println!();
        }
    }

    #[test]
    fn inactive_examples_iter() {
        let v = vec![2,3,5,6,9];
        let iter = InactiveExampleIter {
            active_examples: &v,
            index: 0,
            n: 11
        };
        assert_eq!(&iter.collect::<Vec<usize>>(), &[0,1,4,7,8,10]);

        let iter = InactiveExampleIter {
            active_examples: &v,
            index: 2,
            n: 5
        };
        assert_eq!(&iter.collect::<Vec<usize>>(), &[4]);

        let iter = InactiveExampleIter {
            active_examples: &v,
            index: 1,
            n: 9
        };
        assert_eq!(&iter.collect::<Vec<usize>>(), &[1,4,7,8]);
    }
}
