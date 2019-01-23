use crate::NumT;
use crate::config::Config;

pub trait Objective {
    fn name(&self) -> &'static str;
    fn get_bias(&self) -> NumT;
    fn get_bounds(&self) -> (NumT, NumT);
    fn gradients(&self) -> &[NumT];
    
    fn pseudo_response(&self, target: NumT, prediction: NumT) -> NumT;
    fn optimize_leaf_values(&self, targets: &[NumT]);
}

pub struct L2 {
    bias: NumT,
    bounds: (NumT, NumT),
}

impl L2 {
    pub fn new(targets: &[NumT], predictions: &[NumT]) -> L2 {
        debug_assert_eq!(targets.len(), predictions.len());
        let n = targets.len();

        let mut min = 1.0/0.0;
        let mut max = -1.0/0.0;
        let mut sum = 0.0;
        for i in 0..n {
            let err = targets[i] - predictions[i];
            min = NumT::min(min, err);
            max = NumT::max(max, err);
            sum += err;
        }
        let bias = sum / n as NumT; // mean
        let range = NumT::min(max - bias, bias - min);
        let min = -1.0 * range;
        let max = 1.0 * range;

        L2 {
            bias,
            bounds: (min, max),
        }
    }
}

impl Objective for L2 {
    fn name(&self) -> &'static str { "L2" }
    fn get_bias(&self) -> NumT { self.bias }
    fn get_bounds(&self) -> (NumT, NumT) { self.bounds }
    fn gradients(&self) -> &[NumT] { unimplemented!(); }

    fn pseudo_response(&self, target: NumT, prediction: NumT) -> NumT {
        unimplemented!()
    }

    fn optimize_leaf_values(&self, targets: &[NumT]) {
        unimplemented!()
    }

}

pub struct L1 {}

pub struct Huber {}

pub struct Binary {}
