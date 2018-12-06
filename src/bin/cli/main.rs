#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

extern crate spdyboost;

unsafe fn print_values64(reg: __m256i, bits: bool) {
    let mut stack: [u64; 4] = [0; 4];
    _mm256_storeu_si256(stack[..].as_mut_ptr() as *mut __m256i, reg);
    if bits { for i in 0..4 { print!("{:064b} ", stack[i]); } }
    else    { for i in 0..4 { print!("{:12} ", stack[i]); } }
    println!();
}

fn main() {
    println!("hello world!");

    println!("{}", std::mem::align_of::<spdyboost::bits::BitBlock>());
    println!("{}", spdyboost::bits::BitBlock::nbytes());

    unsafe { 
    let mut m = _mm256_set_epi64x(2, 1, 2, 1);
    print_values64(m, true);
    m = _mm256_permute4x64_epi64(m, 0b01011010);
    print_values64(m, true);
    }
}
