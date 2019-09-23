/*
 * Copyright 2019 DTAI Research Group - KU Leuven.
 * License: Apache License 2.0
 * Author: Laurens Devos
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


