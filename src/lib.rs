extern crate log;
extern crate rand;
extern crate num;
extern crate fnv;

macro_rules! try_or_str {
    ($result:expr, $($arg:tt)*) => {{
        match $result {
            Ok(x) => x,
            Err(_) => return Err(format!($($arg)*))
        }
    }}
}

pub type NumT = f32; // numeric type
pub type NomT = u16; // nominal type

pub mod config;
pub mod dataset;
pub mod bits;
pub mod tree;

//extern crate flate2;
//
//
////extern crate serde;
////#[macro_use] extern crate serde_derive;
////extern crate toml;
//
//pub type Float = f32;
//
//pub mod conf;
//pub mod dataset;
//pub mod bits;
//pub mod tree;
