use crate::NumT;
use crate::config::Config;

use crate::quantile::{ApproxQuantileStats, ApproxQuantile};

pub trait Objective {
    fn name(&self) -> &'static str;
    fn get_bias(&self) -> NumT; // TODO refactor bias
    fn get_bounds(&self) -> (NumT, NumT);
    fn gradients(&self) -> &[NumT];
    
    fn initialize(&mut self, targets: &[NumT], predictions: &[NumT]);

    fn predict_leaf_value(&mut self, targets: &[NumT], examples: &[usize]) -> NumT;
}






// ------------------------------------------------------------------------------------------------

pub struct L2 {
    gradients: Vec<NumT>,
    bias: NumT,
    bounds: (NumT, NumT),
}

impl L2 {
    pub fn new() -> L2 {
        L2 {
            gradients: Vec::new(),
            bias: 0.0,
            bounds: (-1.0, 1.0),
        }
    }
}

impl Objective for L2 {
    fn name(&self) -> &'static str { "L2" }
    fn get_bias(&self) -> NumT { self.bias }
    fn get_bounds(&self) -> (NumT, NumT) { self.bounds }
    fn gradients(&self) -> &[NumT] { &self.gradients }

    fn initialize(&mut self, targets: &[NumT], predictions: &[NumT]) {
        assert_eq!(targets.len(), predictions.len());
        let n = targets.len();
        self.gradients.resize(n, 0.0);

        let mut min = 1.0/0.0;
        let mut max = -1.0/0.0;
        let mut sum = 0.0;
        for i in 0..n {
            let err = targets[i] - predictions[i];
            min = NumT::min(min, err);
            max = NumT::max(max, err);
            sum += err;
            self.gradients[i] = -err;
        }
        let bias = sum / n as NumT; // mean
        let range = NumT::min(max - bias, bias - min);
        let min = -1.0 * range;
        let max = 1.0 * range;

        self.bias = bias;
        self.bounds = (min, max);
    }

    fn predict_leaf_value(&mut self, targets: &[NumT], examples: &[usize]) -> NumT {
        let mut sum = 0.0;
        for &i in examples {
            sum += targets[i];
        }
        sum / examples.len() as NumT // predict mean
    }
}






// ------------------------------------------------------------------------------------------------

pub struct L1 {
    quantile_est: ApproxQuantile,
    gradients: Vec<NumT>,
    bias: NumT,
}

impl L1 {
    pub fn new() -> L1 {
        L1 {
            quantile_est: ApproxQuantile::new(),
            gradients: Vec::new(),
            bias: 0.0,
        }
    }
}

impl Objective for L1 {
    fn name(&self) -> &'static str { "L1" }
    fn get_bias(&self) -> NumT { self.bias }
    fn get_bounds(&self) -> (NumT, NumT) { (-1.0, 1.0) }
    fn gradients(&self) -> &[NumT] { &self.gradients }

    fn initialize(&mut self, targets: &[NumT], predictions: &[NumT]) {
        assert_eq!(targets.len(), predictions.len());
        let n = targets.len();
        let mut stats = ApproxQuantileStats::new();
        self.gradients.resize(n, 0.0);
        
        for i in 0..n {
            let err = targets[i] - predictions[i];
            stats.feed(err);
            self.gradients[i] = -err.signum();
        }

        self.quantile_est.set_stats(&stats);
    }

    fn predict_leaf_value(&mut self, targets: &[NumT], examples: &[usize]) -> NumT {
        self.quantile_est.reset();
        for &i in examples {
            self.quantile_est.feed(targets[i]);
        }
        let mut stage2 = self.quantile_est.stage2(0.5);
        for &i in examples {
            stage2.feed(targets[i]);
        }

        stage2.get_approx_quantile()
    }
}





// ------------------------------------------------------------------------------------------------

pub struct Huber {}






// ------------------------------------------------------------------------------------------------

pub struct Binary {
    gradients: Vec<NumT>,
    bounds: (NumT, NumT),

}

impl Objective for Binary {
    fn name(&self) -> &'static str { "Binary" }
    fn get_bias(&self) -> NumT { 0.0 }
    fn get_bounds(&self) -> (NumT, NumT) { self.bounds }
    fn gradients(&self) -> &[NumT] { &self.gradients }

    fn initialize(&mut self, targets: &[NumT], predictions: &[NumT]) {
        unimplemented!()
    }

    fn predict_leaf_value(&mut self, targets: &[NumT], examples: &[usize]) -> NumT {
        unimplemented!()
    }
}
