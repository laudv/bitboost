use std::str::FromStr;
use crate::NumT;

#[allow(unused_macros)]
macro_rules! parse_config {
    ($cname:ident, $($x:ident: $t:ty = $d:expr, $parser:ident;)*) => {
        #[derive(Debug)]
        pub struct $cname {
            $( pub $x: $t, )*
        }

        impl $cname {
            pub fn new() -> $cname {
                $cname {
                    $( $x: $d, )*
                }
            }

            #[allow(dead_code)]
            pub fn parse<'a, I>(input: I) -> Result<$cname, String>
            where I: 'a + Iterator<Item = &'a str>{
                let mut config = Self::new();

                for line in input {
                    let (name, value) = {
                        let mut iter = line.split('=');
                        let name = iter.next().ok_or(format!("invalid config line: {}", line))?;
                        let value = iter.next().ok_or(format!("invalid config line: {}", line))?;
                        (name.trim(), value.trim())
                    };

                    match name {
                        $( stringify!($x) => {
                            config.$x = $parser(value)
                                .ok_or(format!("config: expected value of type `{}` in line `{}`",
                                               stringify!($t), line))?;
                        },)*
                        _ => { return Err(format!("unknown config value: {}", name)); }
                    }
                }

                Ok(config)
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
    train: String = String::new(),                  parse_fromstr;
    test: String = String::new(),                   parse_fromstr;
    target_feature: isize = -1,                     parse_fromstr;

    objective: String = String::from("L2"),         parse_fromstr;
    metrics: Vec<String> = vec![],                  parse_vec;
    metric_frequency: usize = 1,                    parse_fromstr;

    csv_has_header: bool = true,                    parse_fromstr;
    csv_delimiter: u8 = b',',                       parse_fromstr;

    categorical_features: Vec<usize> = vec![],      parse_vec;

    niterations: usize = 100,                       parse_fromstr;
    learning_rate: NumT = 1.0,                      parse_fromstr;
    reg_lambda: NumT = 0.0,                         parse_fromstr;
    min_examples_leaf: u32 = 1,                     parse_fromstr;
    min_gain: NumT = 1e-6,                          parse_fromstr;
    huber_alpha: NumT = 0.0,                        parse_fromstr;

    max_nbins: usize = 16,                          parse_fromstr;
    discr_nbits: usize = 4,                         parse_fromstr;
    max_tree_depth: usize = 6,                      parse_fromstr;
    compression_threshold: NumT = 0.5,              parse_fromstr;

    random_seed: u64 = 1,                           parse_fromstr;
    feature_fraction: NumT = 1.0,                   parse_fromstr;
    example_fraction: NumT = 1.0,                   parse_fromstr;
    sample_freq: usize = 1,                         parse_fromstr;

    prediction_len: usize = 0,                      parse_fromstr;
);








// ------------------------------------------------------------------------------------------------

mod test {
    use super::*;

    parse_config!(Config2,
        a: i32 = -30,  parse_fromstr;
        b: u32 = 2,    parse_fromstr;
        c: Vec<isize> = vec![], parse_vec;
    );

    #[test]
    fn test() {
        let c = Config2::new();
        assert_eq!(c.a, -30);
        assert_eq!(c.b, 2);
        assert_eq!(&c.c, &[]);

        let args = ["a=12", "b =  13", "c=1,   -2, 3"];
        let c = Config2::parse(args.iter().map(|&s| s)).unwrap();
        assert_eq!(c.a, 12);
        assert_eq!(c.b, 13);
        assert_eq!(&c.c, &[1,-2,3]);
    }
}
