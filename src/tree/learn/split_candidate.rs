/*
 * Copyright (c) DTAI - KU Leuven - All rights reserved.
 * Proprietary, do not copy or distribute without permission.
 * Written by Laurens Devos, 2019
*/

use std::sync::Arc;

use crate::bitblock::BitBlocks;
use crate::bitset::Bitset;

pub struct SplitCandidate {
    split_set: Arc<Bitset>,
    indexes: Arc<BitBlocks>,
}

impl SplitCandidate {
    pub fn is_compressed(&self) -> bool {
        !self.indexes.is_empty()
    }
}


