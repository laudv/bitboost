use std::ops::Sub;

use dataset::{Dataset, FeatureRepr};

pub type SliceRange = (u32, u32);

pub struct SliceStore<T>
where T: Default + Clone {

    /// Storage for slices.
    buffer: Vec<T>,

    /// Freed slices: very simple design, but allows simple memory reuse w/o reallocation.
    free: Vec<SliceRange>,
}

impl <T> SliceStore<T>
where T: Default + Clone {

    pub fn new(initial_cap: usize) -> Self {
        SliceStore {
            buffer: Vec::with_capacity(initial_cap),
            free: Vec::new(),
        }
    }

    pub fn get_slice(&self, r: SliceRange) -> &[T] {
        &self.buffer[r.0 as usize..r.1 as usize]
    }

    pub fn get_slice_mut(&mut self, r: SliceRange) -> &mut [T] {
        &mut self.buffer[r.0 as usize..r.1 as usize]
    }

    pub fn alloc_slice(&mut self, len: u32) -> SliceRange {
        // check if we can use a free one
        for i in 0..self.free.len() {
            let r = self.free[i];
            if r.1 - r.0 >= len {
                self.free.swap_remove(i);
                return r;
            }
        }

        // allocate new memory in buffer
        let old_len = self.buffer.len();
        let new_len = old_len + len as usize;
        assert!(new_len < u32::max_value() as usize);
        self.buffer.resize(new_len, T::default());
        (old_len as u32, new_len as u32)
    }

    pub fn free_slice(&mut self, range: SliceRange) {
        self.free.push(range)
    }
}

pub struct HistStore<T>
where T: Default + Clone {
    slice_store: SliceStore<T>,
    hist_layout: Vec<u32>,
}

impl <T> HistStore<T>
where T: Default + Clone {
    pub fn new<I>(bin_size_iter: I) -> Self
    where I: Iterator<Item = u32> {
        let mut hist_layout = Vec::new();
        let mut accum = 0;

        hist_layout.push(0);
        for bin_size in bin_size_iter {
            accum += bin_size;
            hist_layout.push(accum);
        }

        HistStore {
            slice_store: SliceStore::new(8192),
            hist_layout: hist_layout,
        }
    }

    pub fn for_dataset(dataset: &Dataset) -> Self {
        // Use the right amount of 'buckets' for each feature
        // Currently only categorical features; one bucket for each cat. feat. value.
        Self::new(dataset.features().iter().map(|f| {
            match f.get_repr() {
                Some(&FeatureRepr::CatFeature(card, _)) => card as u32,
                Some(&FeatureRepr::BitVecFeature(ref bitvecs)) => bitvecs.len() as u32,
                _ => panic!("feat repr not supported by histogram"),
            }
        }))
    }

    fn get_histogram_range(&self, feat_id: usize) -> (usize, usize) {
        let lo = self.hist_layout[feat_id];
        let hi = self.hist_layout[feat_id + 1];
        (lo as usize, hi as usize)
    }

    pub fn get_hist(&self, hists_range: SliceRange, feat_id: usize) -> &[T] {
        let (lo, hi) = self.get_histogram_range(feat_id);
        &self.slice_store.get_slice(hists_range)[lo..hi]
    }

    pub fn get_hist_mut(&mut self, hists_range: SliceRange, feat_id: usize) -> &mut [T] {
        let (lo, hi) = self.get_histogram_range(feat_id);
        &mut self.slice_store.get_slice_mut(hists_range)[lo..hi]
    }
    
    pub fn hists_subtract(&mut self, parent_range: SliceRange, left_range: SliceRange,
                          right_range: SliceRange)
    where T: Sub<Output=T>
    {
        let (plo, phi) = parent_range;
        let (llo, lhi) = left_range;
        let (rlo, rhi) = right_range;

        let buffer = &mut self.slice_store.buffer;

        debug_assert_eq!(phi-plo, *self.hist_layout.last().unwrap());

        for i in 0..(phi-plo) {
            let parent = buffer[(plo+i) as usize].clone();
            let left   = buffer[(llo+i) as usize].clone();
            buffer[(rlo+i) as usize] = parent - left;
        }
    }

    pub fn alloc_hists(&mut self) -> SliceRange {
        let total_bins = *self.hist_layout.last().unwrap();
        self.slice_store.alloc_slice(total_bins)
    }

    pub fn free_hists(&mut self, r: SliceRange) {
        self.slice_store.free_slice(r);
    }
}
