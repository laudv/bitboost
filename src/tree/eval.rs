use log::debug;

use bits::{BitSet, ScaledBitSlice};
use conf::Config;
use dataset::NumericalType;
use tree::loss::{LossFun, LossFunGrad, LossFunHess, LossFunHessConst};

pub type ApproxTarget = ScaledBitSlice<NumericalType>;

pub struct EvalValue {
    /// The optimal value for a certain leaf.
    pub optimal_value: NumericalType,

    /// The contribution to the global loss function of this particular leaf.
    pub loss: NumericalType,

    /// The number of examples in the leaf.
    pub example_count: u64,
}

pub trait SplitEvaluator {

    /// Different split evaluators use different representations of the target values. This
    /// associated type defines this represenation for a particular split evaluator.
    type EvaluationData;

    /// The type used to indicate which examples sort a particular leaf node.
    type ExampleSelector;

    /// Given the raw target values, the previous predictions; produce the evaluation data that is
    /// used by this evaluator.
    fn convert_target_values<I1, I2>(&self, nvalues: usize, target_values: I1, prev_pred: I2)
        -> ApproxTarget
    where I1: Iterator<Item = NumericalType>,
          I2: Iterator<Item = NumericalType>;

    /// Evaluates a split in the left and the right leaf.
    fn eval_split(&self, data: &Self::EvaluationData,
                  parent_examples: &Self::ExampleSelector,
                  left_examples: &Self::ExampleSelector)
        -> (EvalValue, EvalValue);
}

/// Evaluate split approximately using the first derivative of the loss function (gradients) and
/// either a constant second derivative (hessian), or a made-up one that scales the summed
/// gradients.
pub struct FirstOrderSplitEvaluator<'a, L: 'a + LossFun> {
    config: &'a Config,
    loss_fun: L,
}

impl <'a, L> FirstOrderSplitEvaluator<'a, L>
where L: 'a + LossFun + LossFunGrad + LossFunHessConst {
    pub fn new(config: &'a Config, loss_fun: L) -> FirstOrderSplitEvaluator<'a, L> {
        FirstOrderSplitEvaluator {
            config: config,
            loss_fun: loss_fun,
        }
    }
}

impl <'a, L> SplitEvaluator for FirstOrderSplitEvaluator<'a, L>
where L: 'a + LossFun + LossFunGrad + LossFunHessConst {
    type EvaluationData = ApproxTarget; // gradients
    type ExampleSelector = BitSet;

    fn convert_target_values<I1, I2>(&self, nvalues: usize, target_values: I1, prev_pred: I2)
        -> ApproxTarget
    where I1: Iterator<Item = NumericalType>,
          I2: Iterator<Item = NumericalType>,
    {
        let gradients = target_values.zip(prev_pred).map(|(t, p)| self.loss_fun.eval_grad(t, p));
        ApproxTarget::new(
            nvalues,
            self.config.target_values_nbits,
            gradients,
            self.config.target_values_limits.0,
            self.config.target_values_limits.1,
        )
    }

    fn eval_split(&self, data: &Self::EvaluationData, parent: &BitSet, left: &BitSet)
        -> (EvalValue, EvalValue)
    {
        let lambda = self.config.reg_lambda;
        let left_count = parent.count_and(left);
        let right_count = parent.true_count() - left_count;

        let left_grad_sum = data.sum_masked_and(parent, left, left_count);
        let right_grad_sum = data.sum_masked_andnot(parent, left, right_count);

        let const_hess = self.loss_fun.get_const_hess();
        let left_hess_sum = const_hess * left_count as NumericalType;
        let right_hess_sum = const_hess * right_count as NumericalType;

        let left_value = -left_grad_sum / (left_hess_sum + lambda);
        let right_value = -right_grad_sum / (right_hess_sum + lambda);

        let left_loss = -0.5 * ((left_grad_sum * left_grad_sum) / (left_hess_sum + lambda));
        let right_loss = -0.5 * ((right_grad_sum * right_grad_sum) / (right_hess_sum + lambda));
        (
            EvalValue { optimal_value: left_value,  loss: left_loss, example_count: left_count },
            EvalValue { optimal_value: right_value, loss: right_loss, example_count: right_count },
        )
    }
}

/// Evaluate split approximately using a first or second order Taylor approximation of the lost
/// function.
pub struct SecondOrderSplitEvaluator<'a, L: 'a + LossFun> {
    _config: &'a Config,
    _loss_fun: L,
}

impl <'a, L> SplitEvaluator for SecondOrderSplitEvaluator<'a, L>
where L: 'a + LossFun + LossFunGrad + LossFunHess {
    type EvaluationData = (ApproxTarget, ApproxTarget); // gradients, hessians
    type ExampleSelector = BitSet;

    fn convert_target_values<I1, I2>(&self, _nvalues: usize, _target_values: I1, _prev_pred: I2)
        -> ApproxTarget
    where I1: Iterator<Item = NumericalType>,
          I2: Iterator<Item = NumericalType>,
    {
        unimplemented!()
    }

    fn eval_split(&self, _data: &Self::EvaluationData, _parent: &BitSet, _left: &BitSet)
        -> (EvalValue, EvalValue)
    {
        unimplemented!()
    }
}
