macro_rules! try_or_str {
    ($result:expr, $msg:expr) => {{
        match $result {
            Ok(x) => x,
            Err(_) => return Err(String::from($msg))
        }
    }}
}

extern crate num;
extern crate rand;
extern crate csv;
extern crate flate2;

extern crate log;

//extern crate serde;
//#[macro_use] extern crate serde_derive;
//extern crate toml;

pub type Float = f32;

pub mod conf;
pub mod dataset;
pub mod bits;
pub mod tree;
