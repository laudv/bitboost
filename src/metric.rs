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
    let y = 2.0 * t - 1.0;
    (1.0 + (-2.0 * y * p).exp()).ln()
});

impl BinaryLoss {
    pub fn new() -> BinaryLoss { BinaryLoss {} }
}

// ------------------------------------------------------------------------------------------------

pub struct BinaryError {}
impl_metric!(BinaryError, eval_one: |_, t: NumT, p: NumT| {
    if t < 0.5 {             // neg target
        if p < 0.0 { 0.0 }   // good prediction
        else       { 1.0 }   // wrong prediction
    } else { // pos target
        if p < 0.0 { 1.0 }   // wrong prediction
        else       { 0.0 }   // good prediction
    }
});

impl BinaryError {
    pub fn new() -> BinaryError { BinaryError {} }
}
