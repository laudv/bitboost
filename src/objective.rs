use crate::NumT;
use crate::config::Config;
use crate::binner::Binner;

pub trait Objective {
    fn name(&self) -> &'static str;
    fn get_bias(&self) -> NumT; // TODO refactor bias
    fn get_bounds(&self) -> (NumT, NumT);
    fn gradients(&self) -> &[NumT];
    
    fn initialize(&mut self, targets: &[NumT], predictions: &[NumT]);

    fn predict_leaf_value(&mut self, targets: &[NumT], examples: &[usize]) -> NumT;
}






// - Least squares --------------------------------------------------------------------------------

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






// - L1 -- least absolute deviation ---------------------------------------------------------------

pub struct L1 {
    bins: Vec<u32>,
    gradients: Vec<NumT>,
    bias: NumT,
    limits: (NumT, NumT),
}

impl L1 {
    pub fn new() -> L1 {
        L1 {
            bins: vec![0; 2048], // fits in L1
            gradients: Vec::new(),
            bias: 0.0,
            limits: (0.0, 0.0),
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
        let (mut min, mut max) = (1.0 / 0.0, -1.0 / 0.0);
        self.gradients.resize(n, 0.0);
        
        for i in 0..n {
            let target = targets[i];
            let err = target - predictions[i];
            min = target.min(min);
            max = target.max(max);
            self.gradients[i] = -err.signum();
        }

        self.limits = (min, max);
    }

    fn predict_leaf_value(&mut self, targets: &[NumT], examples: &[usize]) -> NumT {
        self.bins.iter_mut().for_each(|x| *x = 0);

        let mut binner = Binner::new(&mut self.bins, self.limits, |x, y| *x += y);
        for &i in examples {
            binner.insert(targets[i], 1);
        }

        let rank = (examples.len() / 2) as u32;
        let (bin, _rank_lo, _rank_hi) = binner.bin_with_rank::<u32, _>(rank, |&x| x);

        // Interpolate between two representatives
        //let value = {
        //    let d = (rank_hi - rank_lo) as NumT;
        //    let u = (rank - rank_lo) as NumT / d;
        //    let r0 = binner.bin_representative(bin);
        //    let r1 = binner.bin_representative(bin + 1);

        //    (1.0 - u) * r0 + u * r1
        //};
        
        // Take the bin representative
        let value = binner.bin_representative(bin);
        value
    }
}





// - Huber loss -----------------------------------------------------------------------------------

pub struct Huber {}






// - Binary log loss ------------------------------------------------------------------------------

pub struct Binary {
    gradients: Vec<NumT>,
    bias: NumT,
}

impl Binary {
    pub fn new() -> Binary {
        Binary {
            gradients: Vec::new(),
            bias: 0.0,
        }
    }
}

impl Objective for Binary {
    fn name(&self) -> &'static str { "Binary" }
    fn get_bias(&self) -> NumT { self.bias }
    fn get_bounds(&self) -> (NumT, NumT) { (-1.0, 1.0) }
    fn gradients(&self) -> &[NumT] { &self.gradients }

    fn initialize(&mut self, targets: &[NumT], predictions: &[NumT]) {
        debug_assert!(targets.iter().all(|&t| t == 0.0 || t == 1.0));
        assert_eq!(targets.len(), predictions.len());
        let n = targets.len();
        self.gradients.resize(n, 0.0);

        for i in 0..n {
            let (t, p) = (targets[i], predictions[i]);
            let y = 2.0 * t - 1.0; // 0.0 -> -1.0; 1.0 -> 1.0
            self.gradients[i] = -(2.0*y) / (1.0 + (2.0*y*p).exp());
        }
    }

    fn predict_leaf_value(&mut self, _: &[NumT], examples: &[usize]) -> NumT {
        let mut num = 0.0;
        let mut den = 0.0;
        for &i in examples {
            let y = -self.gradients[i];
            let yabs = y.abs();
            num += y;
            den += yabs * (2.0 - yabs);
        }
        0.5 * ((num / den) + 1.0)
    }
}
