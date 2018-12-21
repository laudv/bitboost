#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use std::mem::{size_of, align_of};

extern crate spdyboost;

use spdyboost::bits::BitBlock;

#[allow(dead_code)]
unsafe fn print_values64(reg: __m256i, bits: bool) {
    let mut stack: [u64; 4] = [0; 4];
    _mm256_storeu_si256(stack[..].as_mut_ptr() as *mut __m256i, reg);
    if bits { for i in 0..4 { print!("{:064b} ", stack[i]); } }
    else    { for i in 0..4 { print!("{:12} ", stack[i]); } }
    println!();
}

fn main() {
    println!("size_of {}", size_of::<[BitBlock; 4]>());
    println!("align_of {}", align_of::<[BitBlock; 4]>());
}
