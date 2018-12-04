use bits::bitblock::{BitBlockOps, BitBlock};

pub struct Bool2BlockIter<I>
where I: Iterator<Item = bool> {
    iter: I
}

impl<I> Bool2BlockIter<I>
where I: Iterator<Item = bool> {
    pub fn new(iter: I) -> Bool2BlockIter<I> {
        Bool2BlockIter { iter: iter }
    }
}

impl<I> Iterator for Bool2BlockIter<I> 
where I: Iterator<Item = bool> {
    type Item = BitBlock;
    fn next(&mut self) -> Option<BitBlock> {
        let mut block: BitBlock = 0;
        let mut iter_empty = true;
        for i in 0..BitBlock::nbits() {
            if let Some(b) = self.iter.next() {
                block = block.set_bit(i, b);
                iter_empty = false;
            }
        }
        if iter_empty { None } else { Some(block) }
    }
}
