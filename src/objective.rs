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
    /// prediction to its bias (-> use initialize_base).
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

pub fn objective_from_name(name: &str) -> Option<Box<dyn Objective>> {
    match name.to_lowercase().as_str() {
        "l2"     => Some(Box::new(L2::new())),
        "l1"     => Some(Box::new(L1::new())),
        "huber"  => Some(Box::new(Huber::new())),
        "binary" => Some(Box::new(Binary::new())),
        "hinge"  => Some(Box::new(Hinge::new())),
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

macro_rules! quantile {
    (@get_value targets: $self:ident, $targets:ident, $i:ident) => {{
        $targets[$i]
    }};
    (@get_value residuals: $self:ident, $targets:ident, $i:ident) => {{
        $targets[$i] - $self.predictions[$i]
    }};
    (of $values:ident: $q:expr, $self:ident, $limits:expr, $targets:ident, $irange:expr) => {{
        let bins = &mut $self.bins;
        bins.iter_mut().for_each(|x| *x = 0);

        let mut binner = Binner::new(bins, $limits);
        let mut count = 0;
        for i in $irange {
            let value = quantile!(@get_value $values: $self, $targets, i);
            count += 1;
            binner.insert(value, 1, |x, y| *x += y);
        }

        let rank = ((count as NumT) * $q).floor() as u32;
        let bin = binner.bin_with_rank(rank, |&x| x);
        binner.bin_representative(bin + 1)
    }}
}

macro_rules! median {
    (of $values:ident: $self:ident, $limits:expr, $targets:ident, $irange:expr) => {{
        quantile!(of $values: 0.5, $self, $limits, $targets, $irange)
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

        let (mut min, mut max) = (POS_INF, NEG_INF);
        for i in 0..n {
            let err = targets[i] - self.predictions[i];
            min = NumT::min(min, err);
            max = NumT::max(max, err);
            self.gradients[i] = -err;
        }

        let bound = NumT::min(min.abs(), max.abs());
        self.bounds = (-bound, bound);
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
    bins: Vec<u32> = vec![0; 1024]
});

impl Objective for L1 {
    impl_simple_obj_methods!(L1, |_: &L1| (-1.0, 1.0));

    fn initialize(&mut self, config: &Config, targets: &[NumT]) {
        let (mut min, mut max) = (POS_INF, NEG_INF);
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
        let (mut min, mut max) = (POS_INF, NEG_INF);
        for i in 0..n {
            let err = targets[i] - self.predictions[i];
            min = err.min(min);
            max = err.max(max);
            self.gradients[i] = -err.signum();
        }
        self.limits = (min, max);
    }

    fn predict_leaf_value(&mut self, targets: &[NumT], examples: &[usize]) -> NumT {
        let iter = examples.iter().cloned();
        let value = median!(of residuals: self, self.limits, targets, iter);
        self.update_predictions(examples, value)
    }
}






// - Huber loss -----------------------------------------------------------------------------------

objective_struct!(Huber {
    alpha: NumT = 0.0,
    delta: NumT = 0.0,
    limits: (NumT, NumT) = (0.0, 0.0),
    bins: Vec<u32> = vec![0; 1024]
});

impl Objective for Huber {
    impl_simple_obj_methods!(Huber, |this: &Huber| (-this.delta, this.delta));

    fn initialize(&mut self, config: &Config, targets: &[NumT]) {
        let (mut min, mut max) = (POS_INF, NEG_INF);
        for &t in targets {
            min = t.min(min);
            max = t.max(max);
        }
        let n = targets.len();
        let bias = median!(of targets: self, (min, max), targets, 0..n);
        self.initialize_base(config, n, bias);
        assert!(0.0 < config.huber_alpha && config.huber_alpha < 1.0); // TODO proper config validation
        self.alpha = config.huber_alpha;
    }

    fn update(&mut self, targets: &[NumT]) {
        let n = targets.len();
        assert_eq!(self.predictions.len(), n);
        assert_eq!(self.gradients.len(), n);

        // determine delta = alpha-quantile { residuals }
        let (mut min, mut max) = (POS_INF, NEG_INF);
        for i in 0..n {
            let err = targets[i] - self.predictions[i];
            min = err.min(min);
            max = err.max(max);
        }
        self.limits = (min, max);
        self.delta = quantile!(of residuals: self.alpha, self, (min, max), targets, 0..n).abs();

        // set the gradients
        for i in 0..n {
            let err = targets[i] - self.predictions[i];
            let err_abs = err.abs();
            if err_abs <= self.delta {
                self.gradients[i] = err;
            } else {
                self.gradients[i] = self.delta * err.signum();
            }
        }
    }

    fn predict_leaf_value(&mut self, targets: &[NumT], examples: &[usize]) -> NumT {
        // find median of residuals
        let rmed = median!(of residuals: self, self.limits, targets, examples.iter().cloned());

        // compute leaf value
        let mut s = 0.0;
        for &i in examples {
            let err = targets[i] - self.predictions[i];
            let d = err - rmed;
            s += d.signum() * NumT::min(self.delta, d.abs());
        }
        let value = rmed + s / (examples.len() as NumT);
        self.update_predictions(examples, value)
    }
}



// - Binary log loss ------------------------------------------------------------------------------

objective_struct!(Binary {
    bound: NumT = 1.25
});

impl Objective for Binary {
    impl_simple_obj_methods!(Binary, |this: &Binary| (-this.bound, this.bound));

    fn initialize(&mut self, config: &Config, targets: &[NumT]) {
        debug_assert!(targets.iter().all(|&t| t == 0.0 || t == 1.0));
        let n = targets.len();

        let nneg = targets.iter().filter(|&x| *x < 0.5).count() as NumT;
        let npos = n as NumT - nneg;
        let avg  = (-1.0 * nneg + 1.0 * npos) / n as NumT;
        let prior = 0.5 * ((1.0 + avg) / (1.0 - avg)).ln();

        self.initialize_base(config, n, prior);
        self.bound = config.binary_gradient_bound;

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




// - Hinge ----------------------------------------------------------------------------------------

objective_struct!(Hinge {});

impl Objective for Hinge {
    impl_simple_obj_methods!(Hinge, |_: &Hinge| (-1.0, 2.0));

    fn initialize(&mut self, config: &Config, targets: &[NumT]) {
        debug_assert!(targets.iter().all(|&t| t == 0.0 || t == 1.0));
        assert_eq!(config.discr_nbits, 2, "Hinge loss requires 2 bits");
        let n = targets.len();

        let nneg = targets.iter().filter(|&x| *x < 0.5).count() as NumT;
        let npos = n as NumT - nneg;
        let prior = (nneg * -1.0 + npos) / n as NumT;

        self.initialize_base(config, n, prior);

        println!("[   ] hinge objective: pos {}, neg {}, prior {}", npos, nneg, prior);
    }

    fn update(&mut self, targets: &[NumT]) {
        let n = targets.len();
        assert_eq!(self.predictions.len(), n);
        assert_eq!(self.gradients.len(), n);

        for i in 0..n {
            let g = &mut self.gradients[i];
            let (t, p) = (targets[i], self.predictions[i]);
            if t < 0.5 { // neg class
                *g = if p > -1.0 { 1.0 } else { 0.0 };
            } else {     // pos class
                *g = if p < 1.0 { -1.0 } else { 0.0 };
            }
        }
    }

    fn predict_leaf_value(&mut self, targets: &[NumT], examples: &[usize]) -> NumT {
        let mut value = 0.0;
        for &i in examples {
            let (t, p) = (targets[i], self.predictions[i]);
            if t < 0.5 { // neg class
                let x = (2.0 + p).max(0.0);
                value -= x*x;
            } else {
                let x = (2.0 - p).max(0.0);
                value += x*x;
            }
        }
        value /= examples.len() as NumT;

        //println!("leaf value = {} #pos={}, #neg={}", value, pcnt, ncnt);
        //for &i in examples.iter().take(10) {
        //    let (t, p) = (targets[i], self.predictions[i]);
        //    println!("{:5} {:2} (g={:2}) {:.3} -> {:.3}", i, t, self.gradients[i], p, p + value);
        //}
        //println!();

        self.update_predictions(examples, value)
    }
}
