use std::marker::PhantomData;

pub mod loss;

use conf::Config;
use bits::BitSet;
use dataset::NumericalType;
use tree::TargetValues;
use objective::loss::{LossFun, LossFunGrad, LossFunHess};

pub struct ObjectiveEval {
    pub left_value: NumericalType,
    pub right_value: NumericalType,
    pub left_loss: NumericalType,
    pub right_loss: NumericalType,
}

impl ObjectiveEval {
    fn new(left_value: NumericalType, right_value: NumericalType,
           left_loss: NumericalType, right_loss: NumericalType) -> ObjectiveEval
    {
        ObjectiveEval {
            left_value: left_value,
            right_value: right_value,
            left_loss: left_loss,
            right_loss: right_loss,
        }
    }
}


/// The objective guides the tree builder in the splitting process by evaluating the gain of
/// candidate splits proposed by the tree builder.
pub trait Objective {
    /// Type used to select examples that sort into left or right node.
    type Sel;

    /// Evaluate the split
    fn eval(&self, left_examples: &Self::Sel, right_examples: &Self::Sel) -> ObjectiveEval;
}


/// Objective function that approximates 
pub struct ApproxObjective<'a, L: 'a + LossFun + LossFunGrad> {
    config: &'a Config,
    gradients: TargetValues,
    hessians: TargetValues,
    _marker: PhantomData<L>,
}

impl <'a, L> ApproxObjective<'a, L>
where L: 'a + LossFun + LossFunGrad {
    //pub fn new...

    fn get_leaf_value_and_local_loss(&self, example_selection: &BitSet)
        -> (NumericalType, NumericalType)
    {
        let grad_sum = self.gradients.sum_masked(example_selection);
        let hess_sum = self.hessians.sum_masked(example_selection);
        let value = -grad_sum / hess_sum;
        let loss = -0.5 * ((grad_sum * grad_sum) / hess_sum);
        (value, loss)
    }
}

impl <'a, L> Objective for ApproxObjective<'a, L>
where L: 'a + LossFun + LossFunGrad + LossFunHess {
    type Sel = BitSet;
    fn eval(&self, left: &BitSet, right: &BitSet) -> ObjectiveEval {
        let (left_value, left_loss) = self.get_leaf_value_and_local_loss(left);
        let (right_value, right_loss) = self.get_leaf_value_and_local_loss(right);
        ObjectiveEval::new(left_value, right_value, left_loss, right_loss)
    }
}





















//pub struct ExactObjective<L> {
//    loss_fun: L,
//    target_values: Vec<NumericalType>,
//}
