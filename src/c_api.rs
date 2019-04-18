use std::panic::catch_unwind;
use std::slice;
use std::alloc::{Layout, alloc, dealloc};
use std::ptr;
use std::ffi::CStr;

use libc::{c_int, c_void, c_char};

use crate::NumT;
use crate::config::Config;
use crate::data::Data;
use crate::tree::AdditiveTree;
use crate::boost::Booster;
use crate::objective::objective_from_name;
use crate::metric::metrics_from_names;


const NEG1: c_int = -1;
const NULL: *mut c_void = 0 as *mut c_void;


struct Context {
    config: Config,
    data: Data,
    model: Option<AdditiveTree>,
}

impl Context {
    unsafe fn from_raw_ptr<'b>(context: *const c_void) -> &'b Context {
        &*(context as *const Context)
    }
    unsafe fn from_raw_ptr_mut<'b>(context: *mut c_void) -> &'b mut Context {
        &mut *(context as *mut Context)
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        println!("Rust: dropping Context {:p}", self as *const Context);
    }
}



macro_rules! wrap_catch_unwind {
    [ ] => {};
    [ 
        $(#[$outer_meta:meta])*
        fn $fname:ident ( $($arg:ident : $type:ty),* )
            -> $rtype:ty | $errval:ident { $($body:tt)* }
        $($other:tt)*
    ] => {
        $(#[$outer_meta])*
        #[no_mangle]
        pub extern fn $fname($($arg : $type),*) -> $rtype {
            let res = catch_unwind(|| {
                $($body)*
            });
            match res {
                Ok(v) => v,
                Err(_) => $errval,
            }
        }

        wrap_catch_unwind!($($other)*);
    }
}

wrap_catch_unwind!(

    /// Get the number of bytes used for a `NumT` value: 4 for single precision float, 8 for double
    /// precision float.
    fn bb_get_numt_nbytes() -> c_int | NEG1 {
        std::mem::size_of::<NumT>() as c_int
    }

    /// Initialize a context. An unmanaged pointer is returned; users of this interface should call
    /// `bb_dealloc` to avoid memory leaks.
    fn bb_alloc(nfeatures: c_int, nexamples: c_int) -> *mut c_void | NULL {
        assert!(nfeatures > 0);
        assert!(nexamples > 0);
        unsafe {
            let layout = Layout::new::<Context>();
            let ptr = alloc(layout) as *mut c_void;

            let config = Config::new();
            let data = Data::empty(&config, nfeatures as usize, nexamples as usize);

            ptr::write(ptr as *mut Context, Context {
                config,
                data,
                model: None,
            });

            //let context = Context::from_raw_ptr_mut(ptr);
            //println!("{:?}", context.config);
            println!("Rust: alloc Context {:?}", ptr);

            ptr
        }
    }

    /// Call this to clean up your `*const Context` pointer.
    fn bb_dealloc(ptr: *mut c_void) -> c_int | NEG1 {
        println!("Rust: dealloc Context {:?}", ptr);
        unsafe {
            let layout = Layout::new::<Context>();
            ptr::drop_in_place(ptr as *mut Context);
            dealloc(ptr as *mut u8, layout);
            0
        }
    }

    fn bb_set_feature_data(ptr: *mut c_void, feat_id: c_int, data: *const NumT, is_categorical: c_int)
        -> c_int | NEG1
    {
        //println!("Rust: set_feature_data({:p}, {}, {:p})", ptr, feat_id, data);
        assert!(feat_id >= 0);
        let feat_id = feat_id as usize;

        unsafe {
            let context = Context::from_raw_ptr_mut(ptr);
            assert!(feat_id <= context.data.nfeatures());
            let data = slice::from_raw_parts(data, context.data.nexamples());
            context.data.set_feature_data(feat_id, data, is_categorical != 0).unwrap();
        }
        0
    }

    fn bb_set_config_field(ptr: *mut c_void, name: *const c_char, value: *const c_char)
        -> c_int | NEG1
    {
        unsafe {
            let name = CStr::from_ptr(name).to_str().expect("invalid utf-8 in key");
            let value = CStr::from_ptr(value).to_str().expect("invalid utf-8 in value");
            let context = Context::from_raw_ptr_mut(ptr);
            context.config.parse_record(name, value).unwrap();
        }
        0
    }

    fn bb_train(ptr: *mut c_void) -> c_int | NEG1 {
        let context = unsafe { Context::from_raw_ptr_mut(ptr) };

        let mut objective = objective_from_name(&context.config.objective)
            .expect("unknown objective");
        let metrics = metrics_from_names(&["l2".to_owned()]).expect("unknown metric");
        let booster = Booster::new(&context.config, &context.data, objective.as_mut(), &metrics);
        context.model = Some(booster.train());
        0
    }

    fn bb_predict(ptr: *mut c_void, result_out: *mut NumT) -> c_int | NEG1 {
        unsafe {
            let context = Context::from_raw_ptr_mut(ptr);
            let model = context.model.as_ref().expect("no trained model");
            let result_out = slice::from_raw_parts_mut(result_out, context.data.nexamples());

            model.predict_buf(&context.data, result_out);
        }
        0
    }
);
