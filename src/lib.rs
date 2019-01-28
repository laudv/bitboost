macro_rules! try_or_str {
    ($result:expr, $($arg:tt)*) => {{
        match $result {
            Ok(x) => x,
            Err(_) => return Err(format!($($arg)*))
        }
    }}
}

pub type NumT = f32; // numeric type
pub const EPSILON: NumT = std::f32::EPSILON;

pub mod config;
pub mod dataset;
pub mod bitblock;
pub mod simd;
pub mod slice_store;
pub mod tree;
pub mod tree_learner;
pub mod objective;
pub mod binner;
pub mod numfeat_mask_store;
pub mod metric;
pub mod boost;
