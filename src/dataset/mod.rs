use std::collections::HashMap;
use std::rc::Rc;

use conf::Config;
use bits::BitSet;

pub enum FeatureData {
    /// Low cardinal nominal data: bit sets for one-hot encoding. Store a bitset for each value.
    BitSets(Vec<(i64, BitSet)>),
}

pub struct Feature {
    id: usize,
    name: String,
    data: FeatureData,
}

pub struct Dataset {
    config: Rc<Config>,
    features: Vec<Feature>,
}





// - Feature impl ---------------------------------------------------------------------------------

impl Feature {
    pub fn get_name(&self) -> &str { &self.name }
    pub fn get_id(&self) -> usize { self.id }
    pub fn get_data(&self) -> &FeatureData { &self.data }

    pub fn set_feature_name(&mut self, name: &str) { self.name = String::from(name); }
    pub fn set_data(&mut self, data: FeatureData) { self.data = data; }
}





// - Dataset impl ---------------------------------------------------------------------------------

impl Dataset {
    pub fn new(config: Rc<Config>) -> Dataset {
        Dataset {
            config: config,
            features: Vec::new(),
        }
    }

    pub fn get_feature(&self, id: usize) -> &Feature { &self.features[id] }
    pub fn get_feature_mut(&mut self, id: usize) -> &mut Feature { &mut self.features[id] }

    /// Add a new low cardinality nominal feature. The feature id is returned.
    pub fn add_lowcard_nominal_feature<I>(&mut self, len: usize, iter: I) -> Result<usize, String>
    where I: Iterator<Item = i64> {
        let mut map: HashMap<i64, BitSet> = HashMap::new();
        
        // Loop over feature data in `iter` and construct bitsets for each possible value
        for (i, v) in iter.enumerate() {
            if !map.contains_key(&v) {
                if map.len() + 1 > self.config.max_lowcard_nominal_cardinality {
                    return Err(format!("low cardinality nominal feature with more than {} distinct
                                       values", self.config.max_lowcard_nominal_cardinality));
                }
                map.insert(v, BitSet::new(len));
            }

            if let Some(bs) = map.get_mut(&v) {
                bs.set_bit(i, true);
            } else {
                return Err(String::from("lowcard feature construction hash error"));
            }
        }
        
        // Construct the feature
        let id = self.features.len();
        let mut bitset_vec: Vec<(i64, BitSet)> = map.into_iter().collect();
        bitset_vec.sort_by(|p, q| p.0.cmp(&q.0));

        let feature = Feature {
            id: id,
            name: String::new(),
            data: FeatureData::BitSets(bitset_vec),
        };

        self.features.push(feature);

        Ok(id)
    }
}











// - Tests ----------------------------------------------------------------------------------------
#[cfg(test)]
mod test {
    use dataset::{Dataset, FeatureData};
    use conf::Config;
    use std::default::Default;
    use std::rc::Rc;

    #[test]
    fn test_add_lowcard_nominal_feature() {
        let mut conf = Config::default();
        conf.max_lowcard_nominal_cardinality = 3;
        let mut dataset = Dataset::new(Rc::new(conf));
        let raw_data = vec![1i64, 2, 1, 1, 2, 2, 2, 3];
        let id = dataset.add_lowcard_nominal_feature(
            raw_data.len(), raw_data.into_iter()).unwrap();

        let feature = dataset.get_feature(id);

        let FeatureData::BitSets(bitsets) = feature.get_data();
        assert_eq!(1i64, bitsets[0].0);
        assert_eq!(2i64, bitsets[1].0);
        assert_eq!(3i64, bitsets[2].0);
        assert_eq!(0b00001101, bitsets[0].1.cast::<u8>()[0]);
        assert_eq!(0b01110010, bitsets[1].1.cast::<u8>()[0]);
        assert_eq!(0b10000000, bitsets[2].1.cast::<u8>()[0]);
    }

    #[test] #[should_panic]
    fn test_add_lowcard_nominal_feature_failure() {
        let mut conf = Config::default();
        conf.max_lowcard_nominal_cardinality = 2;
        let mut dataset = Dataset::new(Rc::new(conf));
        let raw_data = vec![1i64, 2, 1, 1, 2, 2, 2, 3];
        dataset.add_lowcard_nominal_feature(raw_data.len(), raw_data.into_iter()).unwrap();
    }
}
