use std::path::Path;
use std::fs::File;

use bitboost::config::Config;

pub fn main() {
    let dir = Path::new(file!()).parent().unwrap();
    let path = dir.join("../../python/bitboost_config.gen.csv");
    let mut file = File::create(path).unwrap();
    Config::write_doc_csv(&mut file).unwrap();
}
