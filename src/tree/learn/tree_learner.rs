/*
 * Copyright 2019 DTAI Research Group - KU Leuven.
 * License: Apache License 2.0
 * Author: Laurens Devos
*/

use crate::NumT;
use crate::bitslice::*;
use crate::tree::learn::TreeLearnerContext;
use crate::tree::learn::NodeToSplit;
use crate::tree::learn::SplitCandidate;

macro_rules! dispatch_on_discr_nbits {
    ($self:expr, $generic_fun:ident; $( $args:expr ),*) => {{
        match $self.ctx.config.discr_nbits {
            1 => $self.$generic_fun::<BitsliceLayout1>($( $args, )*),
            2 => $self.$generic_fun::<BitsliceLayout2>($( $args, )*),
            4 => $self.$generic_fun::<BitsliceLayout4>($( $args, )*),
            8 => $self.$generic_fun::<BitsliceLayout8>($( $args, )*),
            _ => panic!("invalid discr_nbits"),
        }
    }}
}


pub struct TreeLearner<'a, 'b>
where 'a: 'b, // a lives longer than b
{
    ctx: &'b TreeLearnerContext<'a>,
    
}

impl <'a, 'b> TreeLearner<'a, 'b> {


    fn count_instances_left(&self, n2s: &NodeToSplit, split_cand: &SplitCandidate) -> u64 {

        match (n2s.is_compressed(), split_cand.is_compressed()) {
            (false, false) => {
                0
            },
            (true, false) => {
                0
            },
            (false, true) => {
                0
            },
            (true, true) => {
                0
            }
        }
    }

    fn sum_gradients_left_aux<L>(&self, n2s: &NodeToSplit, split_cand: &SplitCandidate)
        -> NumT
    where L: BitsliceLayout 
    {
        0.0
    }

    fn sum_gradients_left(&self, n2s: &NodeToSplit, split_cand: &SplitCandidate) -> NumT {
        dispatch_on_discr_nbits!(self, sum_gradients_left_aux; n2s, split_cand)
    }
}


