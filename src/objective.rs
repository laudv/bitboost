use crate::{NumT, EPSILON, POS_INF, NEG_INF};
use crate::config::Config;
use crate::binner::Binner;

pub trait Objective {
    fn name(&self) -> &'static str;

    /// The gradient values: the tree learner uses these to build the tree with L2 loss, regardless
    /// of what the objective is. This equals the negative pseudo-residuals (Friedman, 2001).
    fn gradients(&self) -> &[NumT];


    /// The current predictions. These must updated automatically as calls to `predict_leaf_value`
    /// occur.
    fn predictions(&self) -> &[NumT];

    /// The bounds for the discretized gradients (min and max value for `gradients`).
    fn bounds(&self) -> (NumT, NumT);

    /// Return the starting value for this objective (e.g. mean for L2, median for L1).
    fn bias(&self) -> NumT;
    
    /// Initialize the objective. This is done once at the beginning. The objective should set each
    /// prediction to its bias.
    fn initialize(&mut self, config: &Config, targets: &[NumT]);

    /// Update the gradients and related state of the objective so the next tree can be built.
    fn update(&mut self, targets: &[NumT]);

    /// Given a selection of examples, predict the optimal leaf value. This should also update the
    /// predictions of the objective.
    fn predict_leaf_value(&mut self, targets: &[NumT], examples: &[usize]) -> NumT;

    /// Update the prediction of an out-of-bag example. In-bag examples are updated by
    /// `predict_leaf_value`.
    fn update_out_of_bag_prediction(&mut self, i: usize, value: NumT);
}

pub fn objective_from_name(name: &str, _config: &Config) -> Option<Box<dyn Objective>> {
    match name.to_lowercase().as_str() {
        "l2"     => Some(Box::new(L2::new())),
        "l1"     => Some(Box::new(L1::new())),
//        "Huber"   => Box::new(Huber::new(config.huber_alpha)),
        "binary" => Some(Box::new(Binary::new())),
        _        => None,
    }
}

macro_rules! impl_simple_obj_methods {
    ($name:ty, $bounds:expr) => {
        fn name(&self) -> &'static str   { stringify!($name) }
        fn gradients(&self) -> &[NumT]   { &self.gradients }
        fn predictions(&self) -> &[NumT] { &self.predictions }
        fn bounds(&self) -> (NumT, NumT) { $bounds(self) }
        fn bias(&self) -> NumT           { self.bias }

        fn update_out_of_bag_prediction(&mut self, i: usize, value: NumT) {
            //println!("update_pred {:4} -> {:.3} out_of_bag", i, value);
            safety_check!(self.predictions[i].is_finite());
            self.predictions[i] += value;
        }
    }
}

macro_rules! objective_struct {
    ($name:ident { $( $field:ident : $type:ty = $init:expr ),* }) => {
        pub struct $name {
            learning_rate: NumT,
            bias: NumT,
            predictions: Vec<NumT>,
            gradients: Vec<NumT>,
            $(
                $field: $type,
            )*
        }

        impl $name {
            pub fn new() -> $name {
                $name {
                    learning_rate: 0.0,
                    bias: 0.0,
                    predictions: Vec::new(),
                    gradients: Vec::new(),
                    $(
                        $field: $init,
                    )*
                }
            }

            fn initialize_base(&mut self, config: &Config, n: usize, bias: NumT) {
                self.learning_rate = config.learning_rate;
                self.bias = bias;
                self.predictions.clear();
                self.gradients.clear();
                self.predictions.resize(n, bias);
                self.gradients.resize(n, 0.0);
            }

            /// Update the current predictions. `predict_leaf_value` should always call this and
            /// use this functions return value (it's scaled by the learning_rate).
            fn update_predictions(&mut self, examples: &[usize], value: NumT) -> NumT {
                let value = self.learning_rate * value;
                for &i in examples {
                    //println!("update_pred {:4} -> {:.3} predict_leaf_value", i, value);
                    self.predictions[i] += value;
                    safety_check!(self.predictions[i].is_finite());
                }
                value
            }
        }
    }
}

macro_rules! median_value {
    (targets: $self:ident, $targets:ident, $i:ident) => {{
        $targets[$i]
    }};
    (errors: $self:ident, $targets:ident, $i:ident) => {{
        $targets[$i] - $self.predictions[$i]
    }}
}

macro_rules! median {
    (of $values:ident: $self:ident, $limits:expr, $targets:ident, $irange:expr) => {{
        let bins = &mut $self.bins;
        let limits = $limits;

        bins.iter_mut().for_each(|x| *x = 0);

        let mut binner = Binner::new(bins, limits);
        let mut count = 0;
        for i in $irange {
            let value = median_value!($values: $self, $targets, i);
            count += 1;
            binner.insert(value, 1, |x, y| *x += y);
        }

        let rank = count / 2;
        let bin = binner.bin_with_rank(rank, |&x| x);
        binner.bin_representative(bin)
    }}
}





// - Least squares --------------------------------------------------------------------------------

objective_struct!(L2 {
    bounds: (NumT, NumT) = (0.0, 0.0)
});

impl Objective for L2 {
    impl_simple_obj_methods!(L2, |this: &L2| this.bounds);

    fn initialize(&mut self, config: &Config, targets: &[NumT]) {
        let n = targets.len();
        let bias = targets.iter().fold(0.0, |x, y| x+y) / n as NumT;
        self.initialize_base(config, n, bias);
    }

    fn update(&mut self, targets: &[NumT]) {
        let n = targets.len();
        assert_eq!(self.predictions.len(), n);
        assert_eq!(self.gradients.len(), n);

        let mut min = POS_INF;
        let mut max = NEG_INF;
        for i in 0..n {
            let err = targets[i] - self.predictions[i];
            min = NumT::min(min, err);
            max = NumT::max(max, err);
            self.gradients[i] = -err;
        }

        self.bounds = (min, max);
    }

    fn predict_leaf_value(&mut self, targets: &[NumT], examples: &[usize]) -> NumT {
        let mut sum = 0.0;
        for &i in examples {
            sum += targets[i] - self.predictions[i];
        }
        let mean = sum / examples.len() as NumT;
        self.update_predictions(examples, mean)
    }
}






// - L1 -- least absolute deviation ---------------------------------------------------------------

objective_struct!(L1 {
    limits: (NumT, NumT) = (0.0, 0.0),
    bins: Vec<u32> = vec![0; 2048]
});

impl Objective for L1 {
    impl_simple_obj_methods!(L1, |_: &L1| (-1.0, 1.0));

    fn initialize(&mut self, config: &Config, targets: &[NumT]) {
        let (mut min, mut max) = (1.0 / 0.0, -1.0 / 0.0);
        for &t in targets {
            min = t.min(min);
            max = t.max(max);
        }
        let n = targets.len();
        let bias = median!(of targets: self, (min, max), targets, 0..n);
        self.initialize_base(config, n, bias);
    }

    fn update(&mut self, targets: &[NumT]) {
        let n = targets.len();
        assert_eq!(self.predictions.len(), n);
        assert_eq!(self.gradients.len(), n);
        let (mut min, mut max) = (1.0 / 0.0, -1.0 / 0.0);
        for i in 0..n {
            let err = targets[i] - self.predictions[i];
            min = err.min(min);
            max = err.max(max);
            self.gradients[i] = -err.signum();
        }
        self.limits = (min, max);
    }

    fn predict_leaf_value(&mut self, targets: &[NumT], examples: &[usize]) -> NumT {
        self.bins.iter_mut().for_each(|x| *x = 0);
        let iter = examples.iter().cloned();
        let value = median!(of errors: self, self.limits, targets, iter);
        self.update_predictions(examples, value)
    }
}






// - Huber loss -----------------------------------------------------------------------------------

// TODO
//objective_struct!(Huber {
//    bounds: (NumT, NumT) = (0.0, 0.0),
//    bins: Vec<u32> = vec![0; 2048 ],
//    rs: Vec<NumT> = Vec::new(),
//    median: NumT = 0.0,
//    quantile: NumT = 0.0,
//    alpha: NumT = 0.0
//});

/*
pub struct Huber {
    bins: Vec<u32>,
    rs: Vec<NumT>,
    predictions: Vec<NumT>,
    gradients: Vec<NumT>,
    median: NumT,
    quantile: NumT,
    alpha: NumT,
    bounds: (NumT, NumT),
}

impl Huber {
    pub fn new(alpha: NumT) -> Huber {
        Huber {
            bins: vec![0; 100],
            rs: Vec::new(),
            predictions: Vec::new(),
            gradients: Vec::new(),
            median: 0.0,
            quantile: 0.0,
            alpha,
            bounds: (0.0, 0.0),
        }
    }
}

impl Objective for Huber {
    impl_simple_obj_methods!(Huber, |this: &Huber| this.bounds);

    fn bias(&self, targets: &[NumT]) -> NumT {
        self.bins.iter_mut().for_each(|x| *x = 0);

        let mut binner = Binner::new(&mut self.bins, self.limits, |x, y| *x += y);
        for &t in targets {
            binner.insert(t, 1);
        }

        let rank = (targets.len() / 2) as u32;
        let (bin, _rank_lo, _rank_hi) = binner.bin_with_rank::<u32, _>(rank, |&x| x);
        binner.bin_representative(bin)
    }

    fn initialize(&mut self, targets: &[NumT], predictions: &[NumT]) {
        assert_eq!(targets.len(), predictions.len());
        let n = targets.len();
        self.rs.resize(n, 0.0);
        self.gradients.resize(n, 0.0);

        let mut min: NumT =  1.0 / 0.0;
        let mut max: NumT = -1.0 / 0.0;

        for i in 0..n {
            let r = targets[i] - predictions[i];
            self.rs[i] = r;
            min = min.min(r);
            max = max.max(r);
        }

        // estimate quantile
        self.bins.iter_mut().for_each(|x| *x = 0);
        let mut binner = Binner::new(&mut self.bins, (min, max), |x, y| *x += y);
        for &r in &self.rs { binner.insert(r, 1); }

        let rank = (n as NumT * self.alpha).round() as u32;
        let (bin, _rank_lo, _rank_hi) = binner.bin_with_rank::<u32, _>(rank, |&x| x);
        self.quantile = binner.bin_representative(bin);
        self.bounds = (-self.quantile.abs(), self.quantile.abs());

        // set gradient values
        for i in 0..n {
            let r = self.rs[i];
            let grad = &mut self.gradients[i];

            if r.abs() <= self.quantile { *grad = -1.0 * r; }
            else { *grad = -1.0 * r.signum() * self.quantile }
        }

        // estimate median
        let rank = (n / 2) as u32;
        let (bin, _rank_lo, _rank_hi) = binner.bin_with_rank::<u32, _>(rank, |&x| x);
        self.median = binner.bin_representative(bin);
    }

    fn predict_leaf_value(&mut self, _: &[NumT], examples: &[usize]) -> NumT {
        let c = (examples.len() as NumT).recip();
        let mut sum = 0.0;
        for &i in examples {
            let rdiff = self.rs[i] - self.median;
            sum += rdiff.signum() * self.quantile.min(rdiff.abs());
        }
        self.median + c * sum
    }
}



*/


// - Binary log loss ------------------------------------------------------------------------------

objective_struct!(Binary { });

impl Objective for Binary {
    impl_simple_obj_methods!(Binary, |_: &Binary| (-1.25, 1.25));

    fn initialize(&mut self, config: &Config, targets: &[NumT]) {
        debug_assert!(targets.iter().all(|&t| t == 0.0 || t == 1.0));
        let n = targets.len();

        let nneg = targets.iter().filter(|&x| *x < 0.5).count() as NumT;
        let npos = n as NumT - nneg;
        let avg  = (-1.0 * nneg + 1.0 * npos) / n as NumT;
        let prior = 0.5 * ((1.0 + avg) / (1.0 - avg)).ln();

        self.initialize_base(config, n, prior);

        println!("[   ] binary objective: pos {}, neg {}, prior {}", npos, nneg, prior);
    }

    fn update(&mut self, targets: &[NumT]) {
        let n = targets.len();
        assert_eq!(self.predictions.len(), n);
        assert_eq!(self.gradients.len(), n);

        for i in 0..n {
            let (t, p) = (targets[i], self.predictions[i]);
            let y = 2.0 * t - 1.0; // 0.0 -> -1.0; 1.0 -> 1.0
            self.gradients[i] = -(2.0 * y) / (1.0 + (2.0 * y * p).exp());
        }
    }

    fn predict_leaf_value(&mut self, _: &[NumT], examples: &[usize]) -> NumT {
        let mut num = 0.0;
        let mut den = EPSILON;
        for &i in examples {
            let y = -self.gradients[i];
            let yabs = y.abs();
            num += y;
            den += yabs * (2.0 - yabs);
        }
        let value = num / den;
        self.update_predictions(examples, value)
    }
}
