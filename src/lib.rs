extern crate log;
extern crate rand;
extern crate num;

macro_rules! try_or_str {
    ($result:expr, $($arg:tt)*) => {{
        match $result {
            Ok(x) => x,
            Err(_) => return Err(format!($($arg)*))
        }
    }}
}

// Type for numbers
pub type NumT = f32;

pub mod config;
pub mod dataset;
pub mod bits;

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
