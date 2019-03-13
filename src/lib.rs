macro_rules! try_or_str {
    ($result:expr, $($arg:tt)*) => {{
        match $result {
            Ok(x) => x,
            Err(_) => return Err(format!($($arg)*))
        }
    }}
}

pub type NumT = f32; // numeric type
#[allow(non_camel_case_types)] pub type NumT_uint = u32; // unsigned int of same size as NumT
pub const EPSILON: NumT = std::f32::EPSILON;
pub const POS_INF: NumT = std::f32::INFINITY;
pub const NEG_INF: NumT = std::f32::NEG_INFINITY;
pub fn into_uint( x: NumT) -> NumT_uint { unsafe { std::mem::transmute::<NumT, NumT_uint>(x) } }
pub fn into_num_t(x: NumT_uint) -> NumT { unsafe { std::mem::transmute::<NumT_uint, NumT>(x) } }

pub mod config;
pub mod dataset;
pub mod data;
pub mod bitblock;
pub mod simd;
pub mod slice_store;
pub mod tree;
pub mod tree_learner;
pub mod objective;
pub mod binner;
pub mod new_binner;
pub mod numfeat_mask_store;
pub mod metric;
pub mod boost;
