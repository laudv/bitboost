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

pub const EPSILON: NumT = std::f32::EPSILON;

pub mod config;
pub mod dataset;
pub mod bitblock;
pub mod simd;
pub mod slice_store;
pub mod tree;
pub mod boost;
pub mod quantile;
