use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::iter::FromIterator;
use std::time::Instant;

use log::{info, warn, debug};

use crate::NumT;

use crate::bitblock::BitBlock;
use crate::config::Config;
use crate::slice_store::{BitBlockStore, BitVecRef, BitVecMut, SliceRange};






// - Feature --------------------------------------------------------------------------------------

pub enum FeatureType {
    Uninitialized,

    /// A regular numerical feature, uses `raw_data` -> NumLt splits
    OrdNum(NumFeatureSplitter),

    /// Ordered categorical feature -> CatLt splits
    OrdCat(FeatureBitVecs),

    /// Unordered categorical feature -> Equality splits
    NomCat(FeatureBitVecs),
}

pub struct Feature {
    /// Linear index of feature.
    id: usize,

    /// Feature type and their additional representations
    feat_type: FeatureType,

    /// Number of split positions (cardinality for cat. features, #split cand. for numerical).
    nbuckets: usize,

    /// The column number of the feature in the dataset.
    colnum: usize,

    /// The name from the header of the dataset.
    name: String,

    /// The raw data from the input file.
    raw_data: Vec<NumT>,
}

impl Feature {
    fn new(colnum: usize, nbuckets: usize, name: String, raw_data: Vec<NumT>) -> Feature {
        Feature {
            id: 0,
            feat_type: FeatureType::Uninitialized,
            nbuckets,
            colnum,
            name,
            raw_data,
        }
    }

    pub fn id(&self) -> usize { self.id }
    pub fn set_id(&mut self, feat_id: usize) { self.id = feat_id }

    pub fn colnum(&self) -> usize { self.colnum }
    pub fn name(&self) -> &str { &self.name }
    pub fn get_value(&self, i: usize) -> NumT { self.raw_data[i] }

    pub fn get_raw_data(&self) -> &[NumT] { &self.raw_data }

    fn get_categories(&self) -> (HashMap<i64, usize>, Vec<NumT>) {
        let mut set = HashSet::<i64>::new();

        for v in &self.raw_data {
            let k = v.round() as i64;
            assert_ne!(set.len(), 0xFFFF);
            set.insert(k);
        }

        let mut keys = set.into_iter().collect::<Vec<i64>>();
        keys.sort();

        // feature value id (fval_id) => rounded feature value
        let value_map = keys.iter().map(|&x| x as NumT).collect::<Vec<NumT>>();

        // rounded feature value => feature value id (fval_id)
        let map = HashMap::from_iter(keys.into_iter().enumerate().map(|(i, k)| (k, i)));

        (map, value_map)
    }

    fn gen_ord_bitvecs(&mut self) {
        unimplemented!()
    }

    fn gen_nom_bitvecs(&mut self) {
        let nexamples = self.raw_data.len();
        let (map, value_map) = self.get_categories();
        let card = map.len();
        let nblocks = BitBlock::blocks_required_for(nexamples);
        let mut store = BitBlockStore::new(card * (nblocks + 1)); // +1 -> avoid expensive realloc

        let mut ranges = Vec::new();
        for _ in 0..card {
            ranges.push(store.alloc_zero_bits(nexamples));
        }

        let mut bitvecs = FeatureBitVecs {
            value_map,
            ranges,
            store,
        };

        for (i, v) in self.raw_data.iter().enumerate() {
            let k = v.round() as i64;
            let feat_val = map[&k];
            let mut bitvec = bitvecs.get_bitvec_mut(feat_val);
            bitvec.set_bit(i, true);
        }

        self.feat_type = FeatureType::NomCat(bitvecs);
        self.nbuckets = card;
    }

    fn gen_num_splitters(&mut self) {
        //self.feat_type = FeatureType::OrdNum(NumFeatureSplitter::new())
        unimplemented!()
    }

    pub fn get_feature_type(&self) -> &FeatureType { &self.feat_type }
    pub fn get_nbuckets(&self) -> usize { self.nbuckets }
}

pub struct FeatureBitVecs {
    value_map: Vec<NumT>, // fval_id -> value
    ranges: Vec<SliceRange>,
    store: BitBlockStore,
}

impl FeatureBitVecs {
    pub fn get_value(&self, fval_id: usize) -> NumT {
        self.value_map[fval_id]
    }

    pub fn get_bitvec(&self, fval_id: usize) -> BitVecRef {
        let range = self.ranges[fval_id];
        self.store.get_bitvec(range)
    }

    fn get_bitvec_mut(&mut self, fval_id: usize) -> BitVecMut {
        let range = self.ranges[fval_id];
        self.store.get_bitvec_mut(range)
    }
}

pub struct NumFeatureSplitter {

}

impl NumFeatureSplitter {

    /// Get bucket representative
    pub fn get_value(&self, bucket_id: usize) -> NumT {
        unimplemented!()
    }
}






// - Dataset --------------------------------------------------------------------------------------

pub struct Dataset {
    nexamples: usize,
    features: Vec<Feature>,
    target: Feature,
}

impl Dataset {
    pub fn from_csv_file(config: &Config, filename: &str) -> Result<Dataset, String> {
        let reader = BufReader::new(try_or_str!(File::open(filename), "cannot open CSV file"));
        Self::from_csv(config, reader)
    }

    pub fn from_csv<R>(config: &Config, mut reader: R) -> Result<Dataset, String>
    where R: BufRead {
        let sep = config.csv_separator;
        let nbuckets = config.nbuckets;
        let mut feature_names = Self::csv_parse_header(config, &mut reader)?;
        let mut raws = Vec::new();
        let start = Instant::now();

        // Loop over lines in file
        let mut line = String::new();
        let mut record_count = 0;
        while let Ok(byte_count) = reader.read_line(&mut line) {
            if byte_count == 0 || line.is_empty() { break }

            record_count += 1;
            if raws.is_empty() { raws.resize(line.split(sep).count(), Vec::new()); }
            for (i, s) in line.split(sep).map(|x| x.trim()).enumerate() {
                let f = try_or_str!(s.parse::<NumT>(), "parse error at record {} col {}: '{}'",
                    record_count, i, line);
                raws[i].push(f);
            }

            line.clear();
        }

        // Find target feature
        let target_id = {
            let t = config.target_feature_id;
            if t < 0 { raws.len().wrapping_sub((-t) as usize) }
            else     { t as usize }
        };
        if target_id >= raws.len() {
            return Err(format!("invalid target feature id: {}", target_id));
        }

        // Construct features
        feature_names.resize(raws.len(), String::new());
        let mut features = raws.into_iter().zip(feature_names).enumerate()
            .map(|(i, (r, n))| Feature::new(i, nbuckets, n, r))
            .collect::<Vec<Feature>>();
        let target = features.remove(target_id);

        // Create linear feature ids
        features.iter_mut().enumerate().for_each(|(i, f)| f.set_id(i));

        let elapsed = start.elapsed();
        info!("Loaded {} features of length {} in CSV format in {:.2} s",
              features.len(), record_count,
              elapsed.as_secs() as f32 + elapsed.subsec_millis() as f32 * 1e-3);
        info!("Target feature: {}, name: '{}'", target_id, target.name());

        let mut dataset = Dataset {
            nexamples: record_count,
            features: features,
            target: target,
        };

        Self::initialize_features(config, &mut dataset)?;
        Ok(dataset)
    }

    fn csv_parse_header<R>(config: &Config, reader: &mut R) -> Result<Vec<String>, String>
    where R: BufRead {
        let sep = config.csv_separator;
        let mut feature_names = Vec::new();
        if !config.csv_has_header { return Ok(feature_names); }

        let mut line = String::new();
        let _len = try_or_str!(reader.read_line(&mut line), "cannot read CSV header line");
        for name in line.split(sep) {
            let owned = name.trim().to_string();
            feature_names.push(owned);
        }

        Ok(feature_names)
    }

    fn initialize_features(config: &Config, dataset: &mut Dataset) -> Result<(), String> {
        let mut ncat = 0;

        // Generate nominal categorical
        for &i in &config.categorical_columns {
            if let Some(feature) = dataset.get_feature_by_colnum_mut(i) {
                feature.gen_nom_bitvecs();
                ncat += 1;
            } else {
                warn!("Unknown feature specified as categorical: {}", i);
            }

        }
        info!("Added {} categorical feature representations", ncat);

        // Remaining unitialized features are numerical
        for feature in dataset.features.iter_mut() {
            match feature.get_feature_type() {
                FeatureType::Uninitialized => {},
                _ => { continue; } // skip all features that are not uninitialized
            }
            feature.gen_num_splitters();
        }

        Ok(())
    }

    pub fn nexamples(&self) -> usize { self.nexamples }
    pub fn nfeatures(&self) -> usize { self.features.len() }
    pub fn features(&self) -> &[Feature] { &self.features }
    pub fn target(&self) -> &Feature { &self.target }

    pub fn get_feature(&self, feat_id: usize) -> &Feature {
        &self.features[feat_id]
    }

    pub fn get_feature_mut(&mut self, feat_id: usize) -> &mut Feature {
        &mut self.features[feat_id]
    }

    fn find_feature_index_by_id(&self, colnum: usize) -> Option<usize> {
        self.features.binary_search_by_key(&colnum, |f| f.id()).ok()
    }

    pub fn get_feature_by_colnum(&self, colnum: usize) -> Option<&Feature> {
        if let Some(i) = self.find_feature_index_by_id(colnum) { Some(&self.features[i]) }
        else { None }
    }

    pub fn get_feature_by_colnum_mut(&mut self, colnum: usize) -> Option<&mut Feature> {
        if let Some(i) = self.find_feature_index_by_id(colnum) { Some(&mut self.features[i]) }
        else { None }
    }

    pub fn get_value(&self, feat_id: usize, example: usize) -> NumT {
        self.get_feature(feat_id).get_raw_data()[example]
    }
}
