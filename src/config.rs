/*
 * Copyright (c) DTAI - KU Leuven - All rights reserved.
 * Proprietary, do not copy or distribute without permission.
 * Written by Laurens Devos, 2019
*/

use std::io::{Write, Error};
use std::str::FromStr;

use crate::NumT;



macro_rules! parse_config {

    ($cname:ident, $($parts:tt)*) => {
        parse_config!(@gen_config $cname, $($parts)*);
        parse_config!(@gen_doc $cname, $($parts)*);
    };

    (@gen_doc $cname:ident, $($(#[$doc:meta])* $x:ident: $t:ty = $d:expr, $parser:ident;)*) => {
        impl $cname {
            #[allow(dead_code)]
            pub fn write_doc_csv<W: Write>(f: &mut W) -> Result<(), Error> {
                writeln!(f, "name,default,type,description")?;
                $(
                    write!(f, "{},", stringify!($x))?;
                    <$t as PythonRepr>::write_repr(&$d, f)?;
                    write!(f, ",\"{}\",", stringify!($t))?;
                    $(
                        let d = stringify!($doc);
                        let l = d.find('"').unwrap() + 2;
                        let r = d.rfind('"').unwrap();
                        write!(f, "\"{}\"", &d[l..r])?;
                    )*
                    writeln!(f)?;
                )*
                Ok(())
            }
        }
    };

    (@gen_config $cname:ident, $($(#[$doc:meta])* $x:ident: $t:ty = $d:expr, $parser:ident;)*) => {
        #[derive(Debug)]
        pub struct $cname {
            $( $(#[$doc])* pub $x: $t, )*
        }

        impl $cname {
            pub fn new() -> $cname {
                $cname {
                    $( $x: $d, )*
                }
            }

            #[allow(dead_code)]
            pub fn parse<'a, I>(lines: I) -> Result<$cname, String>
            where I: 'a + Iterator<Item = &'a str> {
                let mut config = Self::new();

                for line in lines {
                    config.parse_record_str(line)?;
                }

                Ok(config)
            }

            fn parse_record_str<'a>(&mut self, line: &'a str) -> Result<(), String> {
                let mut iter = line.split('=');
                let name = iter.next().ok_or(format!("invalid config line: {}", line))?;
                let value = iter.next().ok_or(format!("invalid config line: {}", line))?;
                self.parse_record(name.trim(), value.trim())?;
                Ok(())
            }

            pub fn parse_record<'a>(&mut self, name: &'a str, value: &'a str) -> Result<(), String> {
                match name {
                    $( stringify!($x) => {
                        self.$x = $parser(value)
                            .ok_or(format!("config: expected value of type `{}` in record `{}={}`",
                                           stringify!($t), name, value))?;
                    },)*
                    _ => { return Err(format!("unknown config field: {}", name)); }
                };
                Ok(())
            }
        }
    }
}

fn parse_fromstr<T: FromStr>(value: &str) -> Option<T> { value.parse::<T>().ok() }
fn parse_vec<T: FromStr>(value: &str) -> Option<Vec<T>> {
    let mut res = Vec::new();
    if value.is_empty() { Some(res) }
    else {
        for v in value.split(',').map(|s| s.trim()) {
            match v.parse::<T>() {
               Ok(x)  => res.push(x),
               Err(_) => return None,
            }
        }
        Some(res)
    }
}








// ------------------------------------------------------------------------------------------------

parse_config!(Config,
    /// Name of training dataset (cli only).
    train: String = String::new(),                  parse_fromstr;

    /// Name of test dataset (cli only).
    test: String = String::new(),                   parse_fromstr;

    /// Name of the objective to use (l2, l1, huber, binary, hinge).
    objective: String = String::from("L2"),         parse_fromstr;

    /// Comma separated list of metrics to evaluate during training (l2, rmse, binary_loss,
    /// binary_error).
    metrics: Vec<String> = vec![],                  parse_vec;

    /// The metrics are evaluated every `metric_frequency` iterations.
    metric_frequency: usize = 1,                    parse_fromstr;

    /// Whether first line of CSV file is header (cli only).
    csv_has_header: bool = true,                    parse_fromstr;
    
    /// Delimiter in CSV data files (cli only).
    csv_delimiter: u8 = b',',                       parse_fromstr;

    /// Comma separated list of categorical feature indexes (starting at 0).
    categorical_features: Vec<usize> = vec![],      parse_vec;

    /// Total number of trees constructed by the model.
    niterations: usize = 100,                       parse_fromstr;

    /// Learning rate / shrinkage: each tree's predictions is multiplied by this value.
    learning_rate: NumT = 1.0,                      parse_fromstr;

    /// L2 regularization parameter.
    reg_lambda: NumT = 0.0,                         parse_fromstr;

    /// Nodes with less than `min_examples_leaf` examples will not be split. 
    min_examples_leaf: u32 = 1,                     parse_fromstr;

    /// Splits need to improve the predictions by at least `min_gain` in order to be executed.
    min_gain: NumT = 1e-6,                          parse_fromstr;

    /// Parameter of Huber loss: bounds of Huber are `huber_alpha` quantiles of gradients.
    huber_alpha: NumT = 0.95,                       parse_fromstr;

    /// Maximum number of bins used during the pre-processing step of numerical and
    /// high-cardinality features. No more than `max_nbins` splits are considered for these kinds
    /// of features.
    max_nbins: usize = 16,                          parse_fromstr;

    /// Number of bits used to discretize the gradients (1, 2, 4, 8).
    discr_nbits: usize = 4,                         parse_fromstr;

    /// Maximum depth of trees.
    max_tree_depth: usize = 6,                      parse_fromstr;

    /// Ratio of (number of zero instance set 32-bit blocks) / (total number of instance set 32-bit
    /// blocks) is compared to `compression_threshold`. If this ratio exceeds
    /// `compression_threshold`, then compression is applied.
    compression_threshold: NumT = 0.5,              parse_fromstr;

    /// Parameter for binary loss. Theoretical bounds for binary log-loss are -2 and 2. More
    /// aggressive settings seem to result in faster convergence.
    binary_gradient_bound: NumT = 1.25,             parse_fromstr;

    /// Random number generation seed (e.g. for bagging, feature sampling).
    random_seed: u64 = 1,                           parse_fromstr;

    /// Fraction of features used by each tree.
    feature_fraction: NumT = 1.0,                   parse_fromstr;

    /// Example fraction / bagging fraction. Fraction of examples used by each tree.
    example_fraction: NumT = 1.0,                   parse_fromstr;

    /// Frequency of re-sampling features/examples; features and examples are sampled every
    /// `sample_freq` iterations. Feature pre-processing also runs at the same moments.
    sample_freq: usize = 1,                         parse_fromstr;
);











// ------------------------------------------------------------------------------------------------

trait PythonRepr {
    fn write_repr<'a, W: Write>(&'a self, f: &mut W) -> Result<(), Error>;
}

impl PythonRepr for String {
    fn write_repr<'a, W: Write>(&'a self, f: &mut W) -> Result<(), Error> {
        write!(f, "\"{}\"", self)
    }
}

impl PythonRepr for bool {
    fn write_repr<'a, W: Write>(&'a self, f: &mut W) -> Result<(), Error> {
        if *self { write!(f, "True") }
        else     { write!(f, "False") }
    }
}

impl <T: PythonRepr> PythonRepr for Vec<T> {
    fn write_repr<'a, W: Write>(&'a self, f: &mut W) -> Result<(), Error> {
        write!(f, "[")?;
        let mut first = true;
        for x in self {
            if !first { write!(f, ",")?; }
            PythonRepr::write_repr(x, f)?;
            first = false;
        }
        write!(f, "]")
    }
}

macro_rules! gen_python_repr {
    ($kind:ident, [ $( $t:ty ), * ]) => { $( gen_python_repr!($kind, $t); )* };
    (owned, $t:ty) => {
        impl PythonRepr for $t {
            fn write_repr<'a, W: Write>(&'a self, f: &mut W) -> Result<(), Error> {
                write!(f, "{}", self)
            }
        }
    }
}

gen_python_repr!(owned, [NumT, usize, u64, u32, u8]);









// ------------------------------------------------------------------------------------------------

mod test {
    use super::*;

    parse_config!(Config2,
        /// doc a
        a: u8 = 1,  parse_fromstr;
        /// doc b
        b: u32 = 2,    parse_fromstr;
        /// doc c
        c: Vec<usize> = vec![], parse_vec;
    );

    #[test]
    fn test() {
        let c = Config2::new();
        assert_eq!(c.a, 1);
        assert_eq!(c.b, 2);
        assert_eq!(&c.c, &[]);

        let args = ["a=12", "b =  13", "c=1,   2, 3"];
        let c = Config2::parse(args.iter().map(|&s| s)).unwrap();
        assert_eq!(c.a, 12);
        assert_eq!(c.b, 13);
        assert_eq!(&c.c, &[1,2,3]);
    }
}
