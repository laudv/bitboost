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
    nfeatures: usize,
    data: Option<Data>,
    model: Option<AdditiveTree>,
}

impl Context {
    #[allow(dead_code)]
    unsafe fn from_raw_ptr<'b>(ptr: *const c_void) -> &'b Context {
        assert!(ptr != ptr::null());
        &*(ptr as *const Context)
    }

    unsafe fn from_raw_ptr_mut<'b>(ptr: *mut c_void) -> &'b mut Context {
        assert!(ptr != ptr::null_mut());
        &mut *(ptr as *mut Context)
    }

    unsafe fn alloc(nfeatures: usize) -> *mut Context {
        let layout = Layout::new::<Context>();
        let ptr = alloc(layout) as *mut c_void;

        let config = Config::new();

        ptr::write(ptr as *mut Context, Context {
            config,
            nfeatures,
            data: None,
            model: None,
        });

        ptr as *mut Context
    }

    unsafe fn dealloc(ptr: *mut Context) {
        assert!(ptr != ptr::null_mut());
        let layout = Layout::new::<Context>();
        ptr::drop_in_place(ptr);
        dealloc(ptr as *mut u8, layout);
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
    fn bb_alloc(nfeatures: c_int) -> *mut c_void | NULL {
        assert!(nfeatures > 0);
        unsafe {
            let ptr = Context::alloc(nfeatures as usize);
            println!("Rust: alloc Context {:?}", ptr);
            ptr as *mut c_void
        }
    }

    /// Call this to clean up your `*const Context` pointer.
    fn bb_dealloc(ptr: *mut c_void) -> c_int | NEG1 {
        println!("Rust: dealloc Context {:?}", ptr);
        unsafe {
            Context::dealloc(ptr as *mut Context);
            0
        }
    }

    /// Allocate memory for data, deallocate previously allocated memory for data.
    fn bb_refresh_data(ptr: *mut c_void, nexamples: c_int) -> c_int | NEG1 {
        assert!(nexamples > 0);
        unsafe {
            let nexamples = nexamples as usize;
            let context = Context::from_raw_ptr_mut(ptr);
            context.data = Some(Data::empty(&context.config, context.nfeatures, nexamples));
        }
        0
    }

    /// Set the data for a single feature.
    fn bb_set_feature_data(ptr: *mut c_void, feat_id: c_int, column: *const NumT,
                           is_categorical: c_int)
        -> c_int | NEG1
    {
        unsafe {
            let context = Context::from_raw_ptr_mut(ptr);
            assert!(0 <= feat_id && feat_id as usize <= context.nfeatures);
            let feat_id = feat_id as usize;
            let data = context.data.as_mut().expect("no data: use bb_refresh_data first");

            let column = slice::from_raw_parts(column, data.nexamples());
            data.set_feature_data(feat_id, column, is_categorical != 0).unwrap();
        }
        0
    }

    /// Set a single config field.
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
    
    /// Train a model on the data.
    fn bb_train(ptr: *mut c_void) -> c_int | NEG1 {
        let context = unsafe { Context::from_raw_ptr_mut(ptr) };

        let data = context.data.as_mut().expect("no data: use bb_refresh_data first");
        let mut objective = objective_from_name(&context.config.objective)
            .expect("unknown objective");
        let metrics = metrics_from_names(&context.config.metrics).expect("unknown metric");
        let booster = Booster::new(&context.config, data, objective.as_mut(), &metrics);
        context.model = Some(booster.train());
        0
    }

    /// Return the predictions of the model.
    fn bb_predict(ptr: *mut c_void, result_out: *mut NumT) -> c_int | NEG1 {
        unsafe {
            let context = Context::from_raw_ptr_mut(ptr);
            let data = context.data.as_mut().expect("no data: use bb_refresh_data first");
            let model = context.model.as_ref().expect("no trained model");
            let result_out = slice::from_raw_parts_mut(result_out, data.nexamples());

            model.predict_buf(data, result_out);
        }
        0
    }
);
