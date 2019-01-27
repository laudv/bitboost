use crate::{NumT, EPSILON};



macro_rules! impl_metric {
    ($type:ty, eval_one: $impl:expr) => {
        impl_metric!($type, eval_all: |this: &$type, targets: &[NumT], preds: &[NumT]| -> NumT {
            let mut loss = 0.0;
            let mut count = 0;
            for (&t, &p) in targets.iter().zip(preds) {
                loss += $impl(this, t, p);
                count += 1;
            }
            loss / count as NumT
        });
    };
    ($type:ty, eval_all: $impl:expr) => {
        impl Metric for $type {
            fn name(&self) -> &'static str { stringify!($type) }
            fn eval(&self, targets: &[NumT], predictions: &[NumT]) -> NumT {
                $impl(self, targets, predictions)
            }
        }
    }
}


pub trait Metric {
    fn name(&self) -> &'static str;
    fn eval(&self, targets: &[NumT], predictions: &[NumT]) -> NumT;
}


// ------------------------------------------------------------------------------------------------

pub struct L2 {}
impl_metric!(L2, eval_one: |_, t, p| {
    let d = t - p;
    d * d
});

impl L2 {
    pub fn new() -> L2 { L2 {} }
}


// ------------------------------------------------------------------------------------------------

pub struct Rmse { l2: L2 }
impl_metric!(Rmse, eval_all: |this: &Rmse, ts: &[NumT], ps: &[NumT]| {
    this.l2.eval(ts, ps).sqrt()
});

impl Rmse {
    pub fn new() -> Rmse { Rmse { l2: L2::new() } }
}


// ------------------------------------------------------------------------------------------------

pub struct BinaryLoss {}
impl_metric!(BinaryLoss, eval_one: |_, t: NumT, p: NumT| {
    if t <= 0.0 { if (1.0 - p) > EPSILON { -(1.0 - p).ln() } else { -EPSILON.ln() } }
    else        { if        p  > EPSILON { -(      p).ln() } else { -EPSILON.ln() } }
});

impl BinaryLoss {
    pub fn new() -> BinaryLoss { BinaryLoss {} }
}

// ------------------------------------------------------------------------------------------------

pub struct BinaryError {}
impl_metric!(BinaryError, eval_one: |_, t: NumT, p: NumT| {
    if (t - p).abs() > 0.5 { 1.0 } else { 0.0 }
});

impl BinaryError {
    pub fn new() -> BinaryError { BinaryError {} }
}
