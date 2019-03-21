use std::io::Read;
use std::path::Path;
use std::fs::File;
use std::cmp::Ordering;
use std::ops::{Range, Deref, DerefMut};

use log::debug;

use rand::{Rng, SeedableRng};
use rand::rngs::SmallRng;

use csv;

use crate::{NumT, CatT, POS_INF, NEG_INF, into_cat, EPSILON};
use crate::config::Config;
use crate::slice_store::{SliceRange, BitBlockStore, BitVecRef};
use crate::binner::Binner;
use crate::simd;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeatType {
    LoCardCat,
    HiCardCat,
    Numerical,
}

/// The last feature is considered the target feature.
pub struct Data {
    max_nbins: usize,
    names: Vec<String>,
    nfeatures: usize,
    nexamples: usize,
    features: Vec<Vec<NumT>>,
    limits: Vec<(NumT, NumT)>, // feature min / max value
    ftypes: Vec<FeatType>,
    cards: Vec<usize>, // only for categorical
}

impl Data {
    pub fn from_csv_path<P: AsRef<Path>>(config: &Config, path: P) -> Result<Data, String> {
        let reader = File::open(path).map_err(|err| format!("path error: {}", err))?;
        Data::from_csv_reader(config, reader)
    }

    pub fn from_csv_reader<R>(config: &Config, mut reader: R) -> Result<Data, String>
    where R: Read
    {
        let mut csv = String::new();
        reader.read_to_string(&mut csv).map_err(|err| format!("read_to_string err: {}", err))?;
        Self::from_csv(config, &csv)
    }

    pub fn from_csv(config: &Config, csv: &str) -> Result<Data, String> {
        let mut record_len = 0;
        let mut record_count = 0;
        let mut features = Vec::<Vec<NumT>>::new();
        let mut limits = Vec::new();
        let mut ftypes = Vec::new();
        let mut cards = Vec::new();
        let mut record = csv::StringRecord::new();
        let mut reader = csv::ReaderBuilder::new()
            .has_headers(config.csv_has_header)
            .delimiter(config.csv_delimiter)
            .from_reader(csv.as_bytes());

        loop {
            match reader.read_record(&mut record) {
                Ok(false) => break,
                Err(err) => return Err(format!("CSV error: {}", err)),
                Ok(true) => {
                    if record_count == 0 {
                        record_len = record.len();
                        features.resize(record_len, Vec::new());
                        limits.resize(record_len, (POS_INF, NEG_INF));
                        ftypes.resize(record_len, FeatType::Numerical);
                        config.categorical_features.iter()
                            .for_each(|&c| if c<record_len { ftypes[c] = FeatType::LoCardCat; });
                        cards.resize(record_len, 0);
                    }

                    for i in 0..record_len {
                        let value = record.get(i)
                            .and_then(|x| x.parse::<NumT>().ok())
                            .ok_or(format!("Parse error at record {}", record_count))?;

                        features[i].push(value);
                        limits[i] = { let l = limits[i]; (l.0.min(value), l.1.max(value)) };
                        if ftypes[i] == FeatType::LoCardCat {
                            if value.round() != value || value < 0.0 {
                                return Err(format!("Invalid categorical value {} at record {}",
                                           value, record_count));
                            }
                            cards[i] = cards[i].max(1 + into_cat(value) as usize);
                        }
                    }
                }
            }
            record_count += 1;
        }

        // update feature types to high cardinality categorical if cards exceeds max_nbins
        for j in 0..record_len {
            if cards[j] > config.max_nbins {
                debug_assert!(ftypes[j] == FeatType::LoCardCat);
                ftypes[j] = FeatType::HiCardCat;
            }
        }

        // extract feature names from header
        let mut names = vec![String::new(); record_len];
        if config.csv_has_header {
            reader.headers()
                .map_err(|err| format!("CSV header error: {}", err))?
                .into_iter()
                .enumerate()
                .for_each(|(i, name)| names[i].push_str(name));
        }

        Ok(Data {
            max_nbins: config.max_nbins,
            names,
            nfeatures: record_len - 1, // last is target
            nexamples: record_count,
            features,
            limits,
            ftypes,
            cards,
        })
    }

    pub fn nfeatures(&self) -> usize { self.nfeatures }
    pub fn nexamples(&self) -> usize { self.nexamples }
    pub fn feat_name(&self, feature: usize) -> &str { &self.names[feature] } // TODO rename
    pub fn feat_limits(&self, feat_id: usize) -> (NumT, NumT) { self.limits[feat_id] } // TODO rename
    pub fn feat_type(&self, feat_id: usize) -> FeatType { self.ftypes[feat_id] } // TODO rename
    pub fn feat_card(&self, feat_id: usize) -> usize { self.cards[feat_id] } // TODO rename
    pub fn target_id(&self) -> usize { self.nfeatures }
    pub fn get_feature(&self, feat_id: usize) -> &[NumT] { &self.features[feat_id] } // TODO rename
    pub fn get_target(&self) -> &[NumT] { &self.features[self.target_id()] } // TODO rename
    pub fn max_nbins(&self, feat_id: usize) -> usize {
        match self.feat_type(feat_id) {
            FeatType::LoCardCat => {
                // binary optimization: only consider one of two options, other goes right anyway
                let nbins = self.feat_card(feat_id);
                if nbins == 2       { 1 } 
                else if nbins == 1  { 0 }
                else                { nbins }
            },
            FeatType::HiCardCat => self.max_nbins,
            FeatType::Numerical => self.max_nbins,
        }
    }

    pub fn is_compatible(&self, other: &Data) -> bool {
        // TODO implement! check if test dataset and training data set are compatible
        // strange issues can occur if they are not
        unimplemented!()
    }
}






// ------------------------------------------------------------------------------------------------



/// A 'Data' struct with all the necessary bitsets for training.
pub struct Dataset<'a> {
    /// The original data as read from the data file.
    data: &'a Data,

    /// Feature sub-selection.
    active_features: Vec<usize>,

    /// Bagging: which rows from 'data' are used in this dataset.
    active_examples: Vec<usize>,

    /// A store owning all bitsets.
    store: BitBlockStore,

    /// Bitsets for each feature, one for each possible value.
    bitvecs: Vec<Vec<SliceRange>>,

    /// For high cardinality categorical features, a list of hashsets is maintained containing all
    /// possible values per split (for an IN-SPLIT). For other features, this list is empty.
    super_categories: Vec<Vec<CatT>>,

    /// For numerical features, store list of possible split values.
    split_values: Vec<Vec<NumT>>,

    /// Bins buffer for quantile approximation using Binner.
    bins_buffer_u32: Vec<u32>,
    bins_buffer_numt: Vec<NumT>,
}

impl <'a> Dataset<'a> {
    pub fn new(data: &'a Data) -> Dataset<'a> {
        Dataset {
            data,
            active_features: Vec::new(),
            active_examples: Vec::new(),
            store: BitBlockStore::new(1024),
            bitvecs: Vec::new(),
            super_categories: Vec::new(),
            split_values: Vec::new(),
            bins_buffer_u32: vec![0; 1024],
            bins_buffer_numt: vec![0.0; 1024],
        }
    }

    // TODO remove
    fn reset(&mut self) {
        self.active_features.clear();
        self.active_examples.clear();
        self.store.reset();
        self.bitvecs.clear();
        self.super_categories.clear();
        self.split_values.clear();
    }

    // TODO rewrite update -- avoid doing expensive calculations when not necessary
    // only update hicard/num features when no feature selection / bagging
    // try to combine this with root node feature selection? -- avoids copying cat features at each
    // iteration
    pub fn update(&mut self, config: &Config, gradient: &[NumT], grad_lims: (NumT, NumT)) {
        self.reset();

        let n = self.data.nexamples();
        let m = self.data.nfeatures();
        let k = ((n as NumT) * config.example_fraction).round() as usize;
        let l = ((m as NumT) * config.feature_fraction).round() as usize;

        // Initializing data structures
        self.active_examples.resize(k, 0);
        self.active_features.resize(l, 0);
        self.bitvecs.resize(m, Vec::new());
        self.super_categories.resize(m, Vec::new());
        self.split_values.resize(m, Vec::new());

        // Bagging and feature sub-selection
        if n == k { self.active_examples.iter_mut().enumerate().for_each(|(i, x)| *x = i); }
        else      { sample(n, &mut self.active_examples, config.random_seed); }
        if m == l { self.active_features.iter_mut().enumerate().for_each(|(i, x)| *x = i); }
        else      { reservoir_sample(m, &mut self.active_features, config.random_seed + 10); }

        // Feature preprocessing
        for u in 0..l {
            let feat_id = self.active_features[u];

            // determine type of feature:
            // [1] low-cardinality categorical = explicitly categorical + card < max_nbins
            // [2] high-cardinality categorical = explicitly categorical + card >= max_nbins
            // [3] numerical = other
            match self.feat_type(feat_id) {
                FeatType::LoCardCat => self.preprocess_locard_cat(feat_id),
                FeatType::HiCardCat => self.preprocess_hicard_cat(feat_id, gradient, grad_lims),
                FeatType::Numerical => self.preprocess_num(feat_id, gradient),
            }
        }
    }

//    pub fn initialize(&mut self) {
//        let nfeatures = self.data.nfeatures();
//        let nexamples = self.data.nexamples();
//        let nactive_features = (nfeatures as NumT * self.config.feature_fraction).round() as usize;
//        let nactive_examples = (nexamples as NumT * self.config.example_fraction).round() as usize;
//
//        self.active_features.extend(0..nactive_features);
//        self.active_examples.extend(0..nactive_examples);
//    }
//
//    pub fn feature_sample(&mut self, seed: u64) {
//        let nfeatures = self.data.nfeatures();
//        assert!(nfeatures > self.nactive_features());
//        reservoir_sample(nfeatures, &mut self.active_features, self.config.random_seed + 11);
//    }
//
//    pub fn example_sample(&mut self, seed: u64) {
//        let nexamples = self.data.nexamples();
//        assert!(nexamples > self.nactive_examples());
//        sample(nexamples, &mut self.active_examples, self.config.random_seed + 82);
//    }

    /// Generate bitsets for each categorical value.
    fn preprocess_locard_cat(&mut self, feat_id: usize) {
        let n = self.nactive_examples();
        let data = self.data.get_feature(feat_id);
        let card = self.data.feat_card(feat_id);
        let iter = self.active_examples.iter().map(|&i| data[i]);
        let bitvecs = construct_bitvecs(&mut self.store, n, card, iter, |x| into_cat(x) as usize);
        self.bitvecs[feat_id] = bitvecs;
    }

    /// - Accumulate gradient mean for each categorical value.
    /// - Sort by accumulated value (-> becomes ordered)
    /// - Generate candidate split values using quantile estimates
    /// - Generate bitsets for IN-SPLITs
    fn preprocess_hicard_cat(&mut self, feat_id: usize, grad: &[NumT], grad_lims: (NumT, NumT)) {
        let n = self.nactive_examples();
        let data = self.data.get_feature(feat_id);
        let card = self.data.feat_card(feat_id);

        // collect gradient sums & counts per category value
        let mut grad_stat_pairs: Vec<(NumT, u32)> = vec![(0.0, 0); card];
        for (i, x) in self.active_examples.iter().map(|&i| data[i]).enumerate() {
            let category = into_cat(x) as usize;
            let entry = &mut grad_stat_pairs[category];
            entry.0 += grad[i];
            entry.1 += 1;
        }

        //dbg!(&grad_lims);

        // accumulate category values: mean -> this determines their ordering
        // combine similar categories using quantile estimations
        self.bins_buffer_u32.iter_mut().for_each(|b| *b = 0);
        let mut binner = Binner::new(&mut self.bins_buffer_u32, grad_lims);
        let combiner = |bin: &mut u32, d: u32| *bin += d;
        for (sum, count) in grad_stat_pairs.iter_mut() {
            if *count != 0 {
                *sum = *sum / *count as NumT;
                binner.insert(*sum, *count, combiner);
            }
        }

        // extract approximate quantiles from bins
        let extractor = |bin: &u32| *bin;
        let rank_step = n as NumT / (self.data.max_nbins + 1) as NumT;
        let ranks = (1..=self.data.max_nbins).map(|i| (i as NumT * rank_step).round() as u32 - 1);
        let qbins = binner.rank_iter(ranks, extractor);
        let mut last_bin = usize::max_value();
        let mut split_weights = Vec::with_capacity(self.data.max_nbins);
        for bin in qbins {
            if bin == last_bin { continue; }
            last_bin = bin;
            split_weights.push(binner.bin_representative(bin + 1));
        }
        let super_card = split_weights.len();
        debug_assert!(super_card <= self.data.max_nbins);

        // generate mapping: category -> super category
        let mut super_categories = vec![0; card];
        for (category, &(mean, _))  in grad_stat_pairs.iter().enumerate() {
            let super_category = split_weights.binary_search_by(|&x| {
                if x < mean { Ordering::Less }
                else { Ordering::Greater }
            }).expect_err("in this universe, nothing is equal (see cmp impl above)");
            super_categories[category] = super_category as CatT;
            //println!("category {:?} -> {:?} [mean {} < {:?}]", category, super_category, mean,
            //         split_weights.get(super_category));
        }

        //dbg!(&split_weights);
        //dbg!(&super_categories);

        // generate bitvecs
        let iter = self.active_examples.iter().map(|&i| data[i]);
        let numt2cat = |x| super_categories[into_cat(x) as usize] as usize;
        let bitvecs = construct_bitvecs(&mut self.store, n, super_card, iter, numt2cat);
        transform_bitvecs_to_ord(&mut self.store, &bitvecs);

        self.bitvecs[feat_id] = bitvecs;
        self.super_categories[feat_id] = super_categories;
    }

    /// - Generate too many split value candidates using quantile estimates.
    /// - Treat the result as a high cardinality categorical 
    fn preprocess_num(&mut self, feat_id: usize, gradient: &[NumT]) {
        let n = self.nactive_examples();
        let data = self.data.get_feature(feat_id);
        let lims = self.data.feat_limits(feat_id);

        // quantile estimation, weighted by gradient values so there is variation in the limited
        // number of split candidates we generate
        self.bins_buffer_numt.iter_mut().for_each(|b| *b = 0.0);
        let mut binner = Binner::new(&mut self.bins_buffer_numt, lims);
        let mut grad_weight_sum = 0.0;
        let combiner = |bin: &mut NumT, d: NumT| *bin += d;
        for (x, t) in self.active_examples.iter().map(|&i| (data[i], gradient[i].abs() + EPSILON)) {
            // XXX Apply weight transformation?
            grad_weight_sum += t;
            binner.insert(x, t, combiner);
        }

        // extract approximate quantiles
        let weight_step = grad_weight_sum / (self.data.max_nbins + 1) as NumT;
        let weights = (1..=self.data.max_nbins).map(|i| i as NumT * weight_step);
        let qbins = binner.rank_iter(weights, |bin| *bin);
        let mut last_bin = usize::max_value();
        let mut split_values = Vec::with_capacity(self.data.max_nbins);
        for bin in qbins {
            if bin == last_bin { continue; }
            last_bin = bin;
            split_values.push(binner.bin_representative(bin + 1));
        }

        //dbg!(&split_values);

        // construct bitvecs
        let iter = self.active_examples.iter().map(|&i| data[i]);
        let numt2cat = |x| {
            split_values.binary_search_by(|&s| {
                if s < x { Ordering::Less }
                else     { Ordering::Greater }
            }).expect_err("in this universe, nothing is equal")
        };
        let bitvecs = construct_bitvecs(&mut self.store, n, split_values.len(), iter, numt2cat);
        transform_bitvecs_to_ord(&mut self.store, &bitvecs);

        self.bitvecs[feat_id] = bitvecs;
        self.split_values[feat_id] = split_values;
    }

    // ----------
    
    /// Get the number of bins actually used (may be less than max_nbins, depending on the
    /// target values).
    pub fn get_nbins(&self, feat_id: usize) -> usize {
        self.bitvecs[feat_id].len()
    }
    
    /// Get the binary representation for a specific split of a feature.
    pub fn get_bitvec(&self, feat_id: usize, split_id: usize) -> BitVecRef {
        let range = self.bitvecs[feat_id][split_id];
        self.store.get_bitvec(range)
    }

    /// Split value: for low-card cat,  tree splits check equality with this value.
    ///              for numerical,     tree splits check lt < with this value.
    ///              for high-card cat, different, compare with super-category! 
    pub fn get_split_value(&self, feat_id: usize, split_id: usize) -> NumT {
        match self.feat_type(feat_id) {
            FeatType::LoCardCat => split_id as NumT,
            FeatType::HiCardCat => split_id as NumT, // == super-category
            FeatType::Numerical => self.split_values[feat_id][split_id],
        }
    }

    /// Get the super-category (category of categories) of a value of a high-cardinality
    /// cateogrical feature.
    pub fn get_super_category(&self, feat_id: usize, value: NumT) -> CatT {
        debug_assert_eq!(self.feat_type(feat_id), FeatType::HiCardCat);
        self.super_categories[feat_id][into_cat(value) as usize]
    }

    pub fn get_super_categories(&self) -> &Vec<Vec<CatT>> {
        &self.super_categories
    }

    pub fn nactive_features(&self) -> usize { self.active_features.len() }
    pub fn active_features(&self) -> &[usize] { &self.active_features }
    pub fn nactive_examples(&self) -> usize { self.active_examples.len() }
    pub fn active_examples(&self) -> &[usize] { &self.active_examples }

    pub fn feat_name(&self, feature: usize) -> &str { &self.data.names[feature] }
    pub fn feat_limits(&self, feat_id: usize) -> (NumT, NumT) { self.data.limits[feat_id] }
    pub fn feat_type(&self, feat_id: usize) -> FeatType { self.data.ftypes[feat_id] }
    pub fn feat_card(&self, feat_id: usize) -> usize { self.data.cards[feat_id] }

    pub fn get_feature(&self, feat_id: usize) -> &[NumT] { self.data.get_feature(feat_id) }
    pub fn get_target(&self) -> &[NumT] { self.data.get_target() }
    pub fn get_data(&self) -> &Data { self.data }
}










/*
// ------------------------------------------------------------------------------------------------

trait ActiveExContainer: Sized {
    fn sample(size: usize, n: usize, rng: &mut SmallRng, sort: bool) -> Self;
    fn resample(&mut self, n: usize, rng: &mut SmallRng, sort: bool);
    fn sampling_enabled(&self) -> bool;
    fn len(&self) -> usize;
    fn _get_unchecked(&self, index: usize) -> usize;
    fn index_iter<'a>(&'a self) -> ActiveExIter<'a, Self> {
        ActiveExIter {
            active_ex: self,
            i: 0,
        }
    }
}

impl ActiveExContainer for Vec<usize> {
    fn sample(size: usize, n: usize, rng: &mut SmallRng, sort: bool) -> Self {
        let mut v = vec![0; size];
        Self::resample(&mut v, n, rng, sort);
        v
    }
    fn resample(&mut self, n: usize, rng: &mut SmallRng, sort: bool) {
        self.iter_mut().for_each(|i| *i = rng.gen_range(0, n));
        if sort { self.sort_unstable(); }
    }
    fn sampling_enabled(&self) -> bool { true }
    fn len(&self) -> usize { Vec::len(self) }
    fn _get_unchecked(&self, index: usize) -> usize {
        unsafe {
            debug_assert!(index < self.len());
            *self.get_unchecked(index)
        }
    }
}

impl ActiveExContainer for Range<usize> {
    fn sample(size: usize, n: usize, _: &mut SmallRng, _: bool) -> Self {
        assert_eq!(size, n);
        0..size
    }
    fn resample(&mut self, _: usize, _: &mut SmallRng, _: bool) {}
    fn sampling_enabled(&self) -> bool { false }
    fn len(&self) -> usize { std::iter::ExactSizeIterator::len(self) }
    fn _get_unchecked(&self, index: usize) -> usize { self.start + index }
}

struct ActiveExIter<'a, T>
where T: ActiveExContainer {
    active_ex: &'a T,
    i: usize,
}

impl <'a, T> Iterator for ActiveExIter<'a, T>
where T: ActiveExContainer {
    type Item = usize;
    fn next(&mut self) -> Option<usize> {
        if self.i < self.active_ex.len() {
            let res = self.active_ex._get_unchecked(self.i);
            self.i += 1;
            Some(res)
        } else {
            None
        }
    }
}





struct GenericDataset<'a, ActiveEx>
where ActiveEx: ActiveExContainer {
    rng: SmallRng,
    config: &'a Config,
    data: &'a Data,

    useful_features: Vec<usize>,
    active_features: Vec<usize>,
    active_examples: ActiveEx,

    store: BitBlockStore,
    bitvecs: Vec<Vec<SliceRange>>,

    supercats: Vec<Vec<CatT>>,
    split_values: Vec<Vec<NumT>>,
}

impl <'a, ActiveEx> GenericDataset<'a, ActiveEx>
where ActiveEx: ActiveExContainer {
    pub fn new(config: &'a Config, data: &'a Data) -> Self {
        let mut rng = SmallRng::seed_from_u64(config.random_seed);
        let mut useful_features = Vec::new();
        let mut store = BitBlockStore::new(2048);

        let nfeatures = data.nfeatures();
        let nexamples = data.nexamples();
        let active_nfeatures = (nfeatures as NumT * config.feature_fraction).round() as usize;
        let active_nexamples = (nexamples as NumT * config.example_fraction).round() as usize;

        let bitvecs         = vec![Vec::new(); nfeatures];
        let supercats       = vec![Vec::new(); nfeatures];
        let split_values    = vec![Vec::new(); nfeatures];
        let active_features = vec![0; active_nfeatures];
        let active_examples = ActiveEx::sample(active_nexamples, nexamples, &mut rng,
                                            config.sort_example_fraction);

        for feat_id in 0..data.nfeatures() {
            let max_nbins = data.max_nbins(feat_id);
            if max_nbins == 0 { continue; } // not a useful feature, only 1 value

            useful_features.push(feat_id);
            bitvecs[feat_id] = (0..max_nbins)
                .map(|_| store.alloc_zero_bits(active_nexamples))
                .collect();
        }

        GenericDataset {
            rng,
            config,
            data,
            useful_features,
            active_features,
            active_examples,
            store,
            bitvecs,
            supercats,
            split_values,
        }
    }

    pub fn data(&self) -> &Data { self.data }
    pub fn is_feature_sampling_enabled(&self) -> bool { self.config.feature_fraction < 1.0 }
    pub fn is_example_sampling_enabled(&self) -> bool { self.active_examples.sampling_enabled() }
    pub fn active_nexamples(&self) -> usize { self.active_examples.len() }

    pub fn update(&mut self, gradient: &[NumT], grad_lims: (NumT, NumT)) {
        if self.is_feature_sampling_enabled() { self.sample_features() }
        if self.is_example_sampling_enabled() { self.sample_examples() }

        self.update_features(gradient, grad_lims);
    }

    fn update_features(&mut self, gradient: &[NumT], grad_lims: (NumT, NumT)) {
        for u in 0..self.active_features.len() {
            let feat_id = self.active_features[u];

            match self.data.feat_type(feat_id) {
                FeatType::LoCardCat => {
                    // if bagging happened, we need to refill bitsets
                    if self.is_example_sampling_enabled() {
                        self.process_locard_cat(feat_id);
                    }
                },
                FeatType::HiCardCat => {
                    self.process_hicard_cat(feat_id, gradient, grad_lims);
                }
                FeatType::Numerical => {
                    self.process_num(feat_id, gradient, grad_lims);
                }
            }
        }
    }


    fn sample_features(&mut self) {
        let n = self.useful_features.len();
        let buffer = &mut self.active_features;

        // sample unique values from 1..n (the number of useful features)
        reservoir_sample(n, buffer, self.config.random_seed + 12);

        // mapping indices to useful features
        for i in self.active_features.iter_mut() {
            *i = self.useful_features[*i]
        }
    }

    fn sample_examples(&mut self)
    where ActiveEx: DerefMut<Target = [usize]> {
        let n = self.data.nexamples();
        let buffer = &mut self.active_examples;
        sample(n, buffer, self.config.random_seed + 82);
    }

    fn process_locard_cat(&mut self, feat_id: usize) {
        self.zero_bitvecs_of_feature(feat_id);

        let bitvecs = &self.bitvecs[feat_id];
        let data = self.data.get_feature(feat_id);

        for (i_local, i_global) in self.active_examples.index_iter().enumerate() {
            let cat = into_cat(data[i_global]) as usize;
            self.store.get_bitvec_mut(bitvecs[cat]).enable_bit(i_local);
        }
    }

    fn process_locard_cat_iter<I>(store: &mut BitBlockStore, bitvecs: &[SliceRange], data: &[NumT],
                                  iter: I)
    where I: Iterator<Item = (usize, usize)>
    {
        for (i_local, i_global) in iter {
            let cat = into_cat(data[i_global]) as usize;
            store.get_bitvec_mut(bitvecs[cat]).enable_bit(i_local);
        }
    }

    fn process_hicard_cat(&mut self, feat_id: usize, gradient: &[NumT], grad_lims: (NumT, NumT)) {
        self.zero_bitvecs_of_feature(feat_id);
    }

    fn process_num(&mut self, feat_id: usize, gradient: &[NumT], grad_lims: (NumT, NumT)) {
        self.zero_bitvecs_of_feature(feat_id);

    }

    fn zero_bitvecs_of_feature(&mut self, feat_id: usize) {
        for &range in &self.bitvecs[feat_id] {
            self.store.get_bitvec_mut(range)
                .cast_mut::<u64>()
                .iter_mut()
                .for_each(|x| *x = 0);
        }
    }
}

impl <'a> GenericDataset<'a, Vec<usize>> {

}

impl <'a> GenericDataset<'a, Range<usize>> {

}

enum InnerDataset<'a> {
    ExampleSamplingEnabled(GenericDataset<'a, Vec<usize>>),
    ExampleSamplingDisabled(GenericDataset<'a, Range<usize>>),
}

pub struct XxxDataset<'a> { // TODO rename
    inner: InnerDataset<'a>
}

impl <'a> XxxDataset<'a> {
    pub fn new(config: &'a Config, data: &'a Data) -> Self {
        if config.example_fraction < 1.0 {
            let dataset = GenericDataset::new(config, data);
            let inner = InnerDataset::ExampleSamplingEnabled(dataset);
            Self { inner }
        } else {
            let dataset = GenericDataset::new(config, data);
            let inner = InnerDataset::ExampleSamplingDisabled(dataset);
            Self { inner }
        }
    }

    pub fn update(&mut self) {

    }
}

*/



// ------------------------------------------------------------------------------------------------

fn sample(n: usize, buffer: &mut [usize], seed: u64) {
    let mut rng: SmallRng = SmallRng::seed_from_u64(seed);
    buffer.iter_mut().for_each(|i| *i = rng.gen_range(0, n));
    buffer.sort_unstable();
}

fn reservoir_sample(n: usize, buffer: &mut [usize], seed: u64) {
    use rand::{Rng, SeedableRng};
    use rand::rngs::SmallRng;

    let mut rng: SmallRng = SmallRng::seed_from_u64(seed);
    let k = buffer.len();

    for i in 0..n {
        if i < k { buffer[i] = i; }
        else {
            let r = rng.gen_range(0, i+1);
            if r < k { buffer[r] = i }
        }
    }
    buffer.sort_unstable();
}

// TODO remove
fn construct_bitvecs<Iter, CatMap>(store: &mut BitBlockStore, nexamples: usize, card: usize,
                                   iter: Iter, numt2cat: CatMap)
    -> Vec<SliceRange>
where Iter: Iterator<Item=NumT>,
      CatMap: Fn(NumT) -> usize,
{
    let mut bitvecs = Vec::with_capacity(card);
    for _ in 0..card { bitvecs.push(store.alloc_zero_bits(nexamples)); }

    for (i, x) in iter.enumerate() {
        let category = (numt2cat)(x) as usize;
        if category < card {
            let mut bitvec = store.get_bitvec_mut(bitvecs[category]);
            bitvec.enable_bit(i);
        }
    }

    bitvecs
}

fn fill_bitvecs<F>(store: &mut BitBlockStore, bitvecs: &[SliceRange], getter: F)
where F: Fn(usize) -> (usize, usize) {
    for i in 0..10 {
        let (index, category) = getter(i);
        let mut bitvec = store.get_bitvec_mut(bitvecs[category]);
        bitvec.enable_bit(index);
    }
}

fn transform_bitvecs_to_ord(store: &mut BitBlockStore, bitvecs: &[SliceRange]) {
    for (&r0, &r1) in bitvecs[0..].iter().zip(bitvecs[1..].iter()) {
        let (bv0, mut bv1) = store.get_two_bitvecs_mut(r0, r1);
        unsafe { simd::or_assign(&mut bv1, &bv0); }
    }
}







// ------------------------------------------------------------------------------------------------

#[cfg(test)]
mod test {
    use crate::config::Config;
    use super::*;

    #[test]
    fn basic_with_header() {
        let mut config = Config::new();
        config.csv_has_header = true;
        config.csv_delimiter = b';';
        config.categorical_features = vec![2];

        let data = Data::from_csv(&config, "a;bb;ccc;t\n1.0;2.0;0.0;0\n4;5;1;1\n\n").unwrap();
        let tindex = data.target_id();

        assert_eq!(data.nfeatures(), 3);
        assert_eq!(data.nexamples(), 2);
        assert_eq!(data.get_feature(0)[0], 1.0);
        assert_eq!(data.get_feature(1)[0], 2.0);
        assert_eq!(data.get_feature(2)[0], 0.0);
        assert_eq!(data.get_feature(0)[1], 4.0);
        assert_eq!(data.get_feature(1)[1], 5.0);
        assert_eq!(data.get_feature(2)[1], 1.0);
        assert_eq!(data.feat_name(0), "a");
        assert_eq!(data.feat_name(1), "bb");
        assert_eq!(data.feat_name(2), "ccc");
        assert_eq!(data.feat_limits(0), (1.0, 4.0));
        assert_eq!(data.feat_limits(1), (2.0, 5.0));
        assert_eq!(data.feat_limits(2), (0.0, 1.0));
        assert_eq!(data.feat_card(0), 0);
        assert_eq!(data.feat_card(1), 0);
        assert_eq!(data.feat_card(2), 2);
        assert_eq!(data.get_feature(tindex)[0], 0.0);
        assert_eq!(data.get_feature(tindex)[1], 1.0);
    }

    #[test]
    fn basic_without_header() {
        let mut config = Config::new();
        config.csv_has_header = false;

        let data = Data::from_csv(&config, "1.0,2.0,3.0,0\n4,5,6,0\n").unwrap();

        assert_eq!(data.nfeatures(), 3);
        assert_eq!(data.nexamples(), 2);
        assert_eq!(data.get_feature(0)[0], 1.0);
        assert_eq!(data.get_feature(1)[0], 2.0);
        assert_eq!(data.get_feature(2)[0], 3.0);
        assert_eq!(data.get_feature(0)[1], 4.0);
        assert_eq!(data.get_feature(1)[1], 5.0);
        assert_eq!(data.get_feature(2)[1], 6.0);
        assert_eq!(data.feat_name(0), "");
        assert_eq!(data.feat_name(1), "");
        assert_eq!(data.feat_name(2), "");
        assert_eq!(data.feat_limits(0), (1.0, 4.0));
        assert_eq!(data.feat_limits(1), (2.0, 5.0));
        assert_eq!(data.feat_limits(2), (3.0, 6.0));
    }
}
