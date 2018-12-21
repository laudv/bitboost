use std::io::{BufRead, BufReader};
use std::fs::File;
use std::time::Instant;

use log::info;

use NumT;
use config::Config;

pub enum FeatureRepr {
    /// A bitslice discretized representation of a feature.
    BitSlice,
}

pub struct Feature {
    id: usize,
    name: String,

    /// The raw data from the input file.
    raw_data: Vec<NumT>,

    /// A representation used by the tree learner.
    reprs: Vec<FeatureRepr>,
}

pub struct Dataset {
    nexamples: usize,
    features: Vec<Feature>,
    target: Feature,
}





// - Feature impl ---------------------------------------------------------------------------------

impl Feature {
    fn new(id: usize, name: String, raw_data: Vec<NumT>) -> Feature {
        Feature {
            id: id,
            name: name,
            raw_data: raw_data,
            reprs: Vec::new(),
        }
    }

    pub fn id(&self) -> usize { self.id }
    pub fn name(&self) -> &str { &self.name }
    pub fn get_value(&self, i: usize) -> NumT { self.raw_data[i] }
}




// - Dataset impl ---------------------------------------------------------------------------------

impl Dataset {
    pub fn from_csv_file(config: &Config, filename: &str) -> Result<Dataset, String> {
        let reader = BufReader::new(try_or_str!(File::open(filename), "cannot open CSV file"));
        Self::from_csv(config, reader)
    }

    pub fn from_csv<R>(config: &Config, mut reader: R) -> Result<Dataset, String>
    where R: BufRead {
        let sep = config.csv_separator;
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
            .map(|(i, (r, n))| Feature::new(i, n, r))
            .collect::<Vec<Feature>>();
        let target = features.remove(target_id);

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

        Self::gen_reprs(config, &mut dataset)?;
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

    fn gen_reprs(config: &Config, _dataset: &mut Dataset) -> Result<(), String> {
        info!("Generating column representations...");

        for i in &config.categorical_columns {
            println!("{} is a cat column", i);
        }

        Ok(())
    }

    pub fn nexamples(&self) -> usize { self.nexamples }
    pub fn nfeatures(&self) -> usize { self.features.len() }
    pub fn features(&self) -> &[Feature] { &self.features }
    pub fn target(&self) -> &Feature { &self.target }

    pub fn get_feature(&self, feat_id: usize) -> Result<&Feature, String> {
        let i = try_or_str!(self.features.binary_search_by_key(&feat_id, |f| f.id()),
                            "feature {} not found", feat_id);
        Ok(&self.features[i])
    }
}
