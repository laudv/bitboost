extern crate spdyboost;
extern crate pretty_env_logger;
extern crate log;

use spdyboost::dataset::{DataSetBuilder, FeatureData, NumericalType};
use spdyboost::conf::Config;
use spdyboost::tree::{TargetValues, TreeBuilder};

pub fn main() -> Result<(), String> {
    pretty_env_logger::init();

    let mut conf = Config::default();
    conf.lowcard_nominal_features = (0..4).collect();
    conf.target_feature = 4;
    conf.max_tree_depth = 3;
    let dataset = DataSetBuilder::from_gzip_csv_file(&conf, "/tmp/data100.csv.gz")?;
    let target_values = if let FeatureData::Numerical(vec) = dataset.get_target_feature().get_data() {
        let max = vec.iter().cloned().fold(0.0/0.0, NumericalType::max);
        let min = vec.iter().cloned().fold(0.0/0.0, NumericalType::min);
        let d = max - min;
        TargetValues::new(vec.len(), 4, vec.iter().cloned(), min + 0.2*d, max - 0.2*d)
    } else { panic!() };

    let mut builder = TreeBuilder::new(&conf, &dataset, &target_values);
    builder.train();

    Ok(())
}
