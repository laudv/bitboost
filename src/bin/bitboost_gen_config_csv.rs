use std::env;
use std::fs::File;
use std::path::Path;

use bitboost::config::Config;

pub fn main() {
    let args = env::args().collect::<Vec<String>>();
    let path = Path::new(args.get(1).expect("must provide path to <settings>.csv as argument"));
    let mut file = File::create(path).expect("could not create <settings>.csv file");
    Config::write_doc_csv(&mut file).expect("could not write <settings.csv file");
}
