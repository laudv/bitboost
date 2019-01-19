use crate::{NumT, EPSILON};

/// A template for loss functions
pub trait LossFun {
    /// Evaluate the loss function.
    fn eval(&self, target_value: NumT, predicted_value: NumT) -> NumT;
    fn lossfun_name<'a>(&'a self) -> &'a str;

    /// Get the optimal minimal and maximal value to consider, and the bias for this loss function
    /// given these (target, predicted) values. The output is (min, max, bias)
    fn boost_stats<I>(&self, mut iter: I) -> (NumT, NumT, NumT)
    where I: Iterator<Item=(NumT, NumT)> { 
        let mut count = 0;
        let mut min = 1.0/0.0;
        let mut max = -1.0/0.0;
        let mut sum = 0.0;
        while let Some((t, p)) = iter.next() {
            let err = t - p;
            min = NumT::min(min, err);
            max = NumT::max(max, err);
            sum += err;
            count += 1;
        }
        //let bias = sum / count as NumT; // mean
        //let range = max - min;
        //let min = min - 0.1 * range;
        //let max = max + 0.1 * range;
        let bias = sum / count as NumT; // mean
        let range = NumT::min(max - bias, bias - min);
        let min = -1.0 * range;
        let max = 1.0 * range;

        (min, max, bias)
    }
}

/// A template for loss functions that have a first derivative
pub trait LossFunGrad: LossFun {
    /// Evaluate first derivative of loss function (gradient).
    fn eval_grad(&self, target_value: NumT, predicted_value: NumT) -> NumT;
}

/// A template for loss functions that have a second derivative
pub trait LossFunHess: LossFunGrad {
    /// Evaluate second derivative of loss function (hessian).
    fn eval_hess(&self, target_value: NumT, predicted_value: NumT) -> NumT;
}

/// A template for loss functions that have a constant second derivative OR there is no second
/// derivative and we simply take the mean of the gradients by having 'hessian = 1.0'.
pub trait LossFunHessConst: LossFunGrad {
    /// Constant second derivative
    fn get_const_hess(&self) -> NumT;
}




macro_rules! impl_eval {
    ($name:ident : $eval:expr) => {
        impl LossFun for $name {
            fn eval(&self, t: NumT, p: NumT) -> NumT { $eval(self, t, p) }
            fn lossfun_name<'a>(&'a self) -> &'a str { stringify!($name) }
        }
    }
}
macro_rules! impl_grad {
    ($name:ident : $grad:expr) => {
        impl LossFunGrad for $name {
            fn eval_grad(&self, t: NumT, p: NumT) -> NumT { $grad(self, t, p) }
        }
    }
}
macro_rules! impl_hess {
    ($name:ident : $hess:expr) => {
        impl LossFunHess for $name {
            fn eval_hess(&self, t: NumT, p: NumT) -> NumT { $hess(self, t, p) }
        }
    };
    ($name:ident : const $hess:expr) => {
        impl LossFunHess for $name {
            fn eval_hess(&self, _: NumT, _: NumT) -> NumT { $hess }
        }
        impl LossFunHessConst for $name {
            fn get_const_hess(&self) -> NumT { $hess }
        }
    }
}




pub struct L2Loss;
impl L2Loss { pub fn new() -> L2Loss { L2Loss { } } }
impl_eval!(L2Loss: |_, t, p| { let d = t-p; d * d });
impl_grad!(L2Loss: |_, t, p| { 2.0*(p-t) });
impl_hess!(L2Loss: const 1.0);

pub struct L1Loss;
impl L1Loss { pub fn new() -> L1Loss { L1Loss { } } }
impl_eval!(L1Loss: |_, t, p| { NumT::abs(p-t) });
impl_grad!(L1Loss: |_, t, p| { NumT::signum(p-t) });
impl_hess!(L1Loss: const 1.0);

pub struct HuberLoss { delta: NumT }
impl HuberLoss { pub fn new(delta: NumT) -> HuberLoss { HuberLoss { delta } } }
impl_eval!(HuberLoss: |s: &HuberLoss, t: NumT, p: NumT| {
    let d = t-p;
    if d.abs() < s.delta { 0.5 * d * d }
    else { s.delta * (d.abs() - 0.5 * s.delta) } });
impl_grad!(HuberLoss: |s: &HuberLoss, t: NumT, p: NumT| {
    let d = t-p;
    if d.abs() < s.delta { d }
    else { d.signum() * s.delta } });
impl_hess!(HuberLoss: const 1.0);

pub struct LogLoss { }
impl LogLoss { 
    pub fn new() -> LogLoss { LogLoss { } }
}
impl_eval!(LogLoss: |_, t: NumT, p: NumT| {
    if t < 0.5 && 1.0 - p > EPSILON { // true label is negative
        -(1.0 - p).ln()
    } else if p > EPSILON {
        -p.ln()
    } else {
        -EPSILON.ln()
    }
});
impl_grad!(LogLoss: |_, t: NumT, p: NumT| {
    let l = t*2.0 - 1.0;
    let resp = -l / (1.0 + (t*p).exp());
    resp
});
