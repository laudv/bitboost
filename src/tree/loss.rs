use dataset::NumericalType as NumT;

/// A template for loss functions
pub trait LossFun {
    /// Evaluate the loss function.
    fn eval(&self, target_value: NumT, predicted_value: NumT) -> NumT;
    fn name() -> &'static str;
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
    ($name:ident : $grad:expr) => {
        impl LossFun for $name {
            fn eval(&self, t: NumT, p: NumT) -> NumT { $grad(self, t, p) }
            fn name() -> &'static str { stringify!($name) }
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
impl_eval!(L2Loss: |_, t, p| { let d = t-p; 0.5 * d * d });
impl_grad!(L2Loss: |_, t, p| { p-t });
impl_hess!(L2Loss: const 1.0);

pub struct L1Loss;
impl L1Loss { pub fn new() -> L1Loss { L1Loss { } } }
impl_eval!(L1Loss: |_, t, p| { NumT::abs(p-t) });
impl_grad!(L1Loss: |_, t, p| { NumT::signum(p-t) });
impl_hess!(L1Loss: const 1.0);

pub struct HuberLoss { delta: NumT }
impl HuberLoss { pub fn new(d: NumT) -> HuberLoss { HuberLoss { delta: d } } }
impl_eval!(HuberLoss: |s: &HuberLoss, t: NumT, p: NumT| {
    let d = t-p;
    if d.abs() < s.delta { 0.5 * d * d }
    else { s.delta * (d.abs() - 0.5 * s.delta) } });
impl_grad!(HuberLoss: |s: &HuberLoss, t: NumT, p: NumT| {
    let d = t-p;
    if d.abs() < s.delta { d }
    else { d.signum() * s.delta } });
impl_hess!(HuberLoss: const 1.0);
