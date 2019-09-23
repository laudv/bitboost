/*
 * Copyright 2019 DTAI Research Group - KU Leuven.
 * License: Apache License 2.0
 * Author: Laurens Devos
*/

use crate::{NumT};



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

pub fn metric_from_name(name: &str) -> Option<Box<dyn Metric>> {
    match name.to_lowercase().as_str() {
        "l2" => Some(Box::new(L2::new())),
        "rmse" => Some(Box::new(Rmse::new())),
        "binaryloss" | "binary_loss" => Some(Box::new(BinaryLoss::new())),
        "binaryerror" | "binary_error" => Some(Box::new(BinaryError::new())),
        "binaryerror01" | "binary_error01" => Some(Box::new(BinaryError01::new())),
        _ => None
    }
}

pub fn metrics_from_names(names: &[String]) -> Option<Vec<Box<dyn Metric>>> {
    let mut metrics = Vec::new();
    for name in names {
        match metric_from_name(name) {
            Some(metric) => metrics.push(metric),
            None => return None,
        }
    }
    Some(metrics)
}

pub fn metric_for_objective(name: &str) -> Option<Box<dyn Metric>> {
    match name.to_lowercase().as_str() {
        "l2" | "l1" | "huber" => Some(Box::new(Rmse::new())),
        "binary" => Some(Box::new(BinaryLoss::new())),
        _ => None
    }
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

// ------------------------------------------------------------------------------------------------

pub struct BinaryError01 {} // use for binary problems with l1/l2 -> choose closest to 0,1 instead of -1, 1
impl_metric!(BinaryError01, eval_one: |_, t: NumT, p: NumT| {
    if t < 0.5 {
        if p < 0.5 { 0.0 }
        else       { 1.0 }
    } else {
        if p < 0.5 { 1.0 }
        else       { 0.0 }
    }
});

impl BinaryError01 {
    pub fn new() -> BinaryError01 { BinaryError01 {} }
}
