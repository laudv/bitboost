/*
 * Copyright (c) DTAI - KU Leuven - All rights reserved.
 * Proprietary, do not copy or distribute without permission.
 * Written by Laurens Devos, 2019
*/

pub mod learn;

// TODO remove
mod learner;
pub use self::learner::{TreeLearner, TreeLearnerContext};

mod tree;
pub use self::tree::{Tree, SplitType, SplitCrit, AdditiveTree};
