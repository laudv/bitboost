/*
 * Copyright 2019 DTAI Research Group - KU Leuven.
 * License: Apache License 2.0
 * Author: Laurens Devos
*/

pub mod learn;

// TODO remove
mod learner;
pub use self::learner::{TreeLearner, TreeLearnerContext};

mod tree;
pub use self::tree::{Tree, SplitType, SplitCrit, AdditiveTree};
