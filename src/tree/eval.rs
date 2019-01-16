use crate::NumT;
use crate::tree::loss::LossFun;

pub trait Evaluator {
    type Input;
    type Output;

    fn evaluator_name<'a>(&'a self) -> &'a str;
    fn evaluate<I1, I2>(&self, targets: I1, predictions: I2) -> NumT
    where I1: Iterator<Item=NumT>,
          I2: Iterator<Item=NumT>;
}

impl <T> Evaluator for T
where T: LossFun {
    type Input = NumT;
    type Output = NumT;

    fn evaluator_name<'a>(&'a self) -> &'a str {
        LossFun::lossfun_name(self)
    }

    fn evaluate<I1, I2>(&self, targets: I1, predictions: I2) -> NumT
    where I1: Iterator<Item=NumT>,
          I2: Iterator<Item=NumT>,
    {
        let mut count = 0;
        let mut accum = 0.0;
        for (t, p) in targets.zip(predictions) {
            accum += self.eval(t, p);
            count += 1;
        }
        accum / count as NumT
    }
}
