/*
 * Copyright 2019 DTAI Research Group - KU Leuven.
 * License: Apache License 2.0
 * Author: Laurens Devos
*/

use std::io::Read;
use std::path::Path;
use std::fs::File;

use csv;

use crate::{NumT, POS_INF, NEG_INF, into_cat};
use crate::config::Config;

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
    pub fn from_csv_path<P>(config: &Config, path: P) -> Result<Data, String>
    where P: AsRef<Path> {
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
                            Self::check_categorical_value(value)?;
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

        let target_id = record_len - 1;
        println!("[   ] using target {} (column {})", names[target_id], target_id);

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

    pub fn empty(config: &Config, nfeatures: usize, nexamples: usize) -> Data {
        let nfeatures = nfeatures + 1; // include target

        let names: Vec<String> = (0..nfeatures).map(|i| format!("feat{:04}", i)).collect();
        let features = vec![vec![0.0; nexamples]; nfeatures];
        let limits = vec![(0.0, 0.0); nfeatures];
        let ftypes = vec![FeatType::Numerical; nfeatures];
        let cards = vec![0; nfeatures];

        Data {
            max_nbins: config.max_nbins,
            names,
            nfeatures: nfeatures - 1, // exclude target
            nexamples,
            features,
            limits,
            ftypes,
            cards,
        }
    }

    pub fn set_feature_data(&mut self, feat_id: usize, data: &[NumT], categorical: bool)
        -> Result<(), String>
    {
        assert_eq!(data.len(), self.nexamples);
        let feat = &mut self.features[feat_id];
        let flim = &mut self.limits[feat_id];
        let card = &mut self.cards[feat_id];
        assert_eq!(feat.len(), self.nexamples);
        for i in 0..self.nexamples {
            let value = data[i];
            feat[i] = value;
            *flim = (flim.0.min(value), flim.1.max(value));

            if categorical {
                Self::check_categorical_value(value)?;
                *card = (*card).max(1 + into_cat(value) as usize);
            }
        }
        if categorical {
            if *card > self.max_nbins {
                self.ftypes[feat_id] = FeatType::HiCardCat;
            } else {
                self.ftypes[feat_id] = FeatType::LoCardCat;
            }
        }
        Ok(())
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

    pub fn is_compatible(&self, _other: &Data) -> bool {
        // TODO implement! check if test dataset and training data set are compatible
        // strange issues can occur if they are not
        unimplemented!()
    }

    fn check_categorical_value(value: NumT) -> Result<(), String> {
        if value.round() != value || value < 0.0 {
            Err(format!("Invalid categorical value {}", value))
        } else {
            Ok(())
        }
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
