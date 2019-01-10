use std::ops::Sub;
use std::fmt::{Debug, Formatter, Result as FmtResult};

use fnv::FnvHashMap as HashMap;
use dataset::{Dataset, FeatureRepr};

const INITIAL_HIST_STORE_CAPACITY: usize = 512;

/// Store a bunch of histograms together in one store. `T` is the information to store per
/// histogram bucket (e.g. gradient sum, hessian sum, bucket count).
pub struct HistStore<T> {
    /// Map a (node_id, feature_id) to a range into the buffer.
    ranges: HashMap<usize, (u32, u32)>,

    /// Storage for histogram data.
    buffer: Vec<T>,

    /// The layout of the histograms => summed cardinalities (how many different buckets per
    /// feature?).
    hist_layout: Vec<u32>,

    /// node_id's whose histograms have been deleted
    recycle_list: Vec<usize>,
}

impl <T> HistStore<T> {
    pub fn new<I>(bin_size_iter: I) -> HistStore<T>
    where I: Iterator<Item = u32> {
        let mut hist_layout = Vec::new();
        let mut accum = 0;

        hist_layout.push(0);
        for bin_size in bin_size_iter {
            accum += bin_size;
            hist_layout.push(accum);
        }

        HistStore {
            ranges: HashMap::default(),
            buffer: Vec::with_capacity(INITIAL_HIST_STORE_CAPACITY),
            hist_layout: hist_layout,
            recycle_list: Vec::new(),
        }
    }

    pub fn for_dataset(dataset: &Dataset) -> HistStore<T> {
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

    fn get_histograms_range(&self, node_id: usize) -> (usize, usize) {
        let (lo, hi) = self.ranges[&node_id];
        (lo as usize, hi as usize)
    }

    fn get_histogram_range(&self, feat_id: usize) -> (usize, usize) {
        let lo = self.hist_layout[feat_id];
        let hi = self.hist_layout[feat_id + 1];
        (lo as usize, hi as usize)
    }

    pub fn get_histogram(&self, node_id: usize, feat_id: usize) -> &[T] {
        let (lo, hi) = self.get_histograms_range(node_id);
        let buffer = &self.buffer[lo..hi];
        let (lo, hi) = self.get_histogram_range(feat_id);
        &buffer[lo..hi]
    }

    pub fn get_histogram_mut(&mut self, node_id: usize, feat_id: usize) -> &mut [T] {
        let (lo1, hi1) = self.get_histograms_range(node_id);
        let (lo2, hi2) = self.get_histogram_range(feat_id);
        let buffer = &mut self.buffer[lo1..hi1];
        &mut buffer[lo2..hi2]
    }

    pub fn get_histograms_mut(&mut self, node_id: usize) -> &mut [T] {
        let (lo, hi) = self.get_histograms_range(node_id);
        &mut self.buffer[lo..hi]
    }

    pub fn histograms_subtract(&mut self, parent_id: usize, left_id: usize, right_id: usize)
    where T: Sub<Output=T> + Copy {
        let (plo, phi) = self.get_histograms_range(parent_id);
        let (llo, _) = self.get_histograms_range(left_id);
        let (rlo, _) = self.get_histograms_range(right_id);

        debug_assert_eq!(phi-plo, *self.hist_layout.last().unwrap() as usize);

        for i in 0..(phi-plo) {
            let diff = self.buffer[plo+i] - self.buffer[llo+i];
            self.buffer[rlo+i] = diff;
        }

    }

    pub fn debug_print(&self, node_id: usize)
    where T: Debug {
        println!("Histograms for node_id {}", node_id);
        for feat_id in 0..self.hist_layout.len()-1 {
            for (i, val) in self.get_histogram(node_id, feat_id).iter().enumerate() {
                println!("{:4}: {:?}", i, val);
            }
            println!();
        }
    }

    pub fn alloc_histograms(&mut self, node_id: usize)
    where T: Default + Clone {
        debug_assert!(!self.ranges.contains_key(&node_id));
        if let Some(freed_node_id) = self.recycle_list.pop() {
            // reuse previous histograms
            let range = self.ranges.remove(&freed_node_id).unwrap();
            self.ranges.insert(node_id, range);
        } else {
            // allocate new ones
            let total_bins = *self.hist_layout.last().unwrap() as usize;
            let old_len = self.buffer.len();
            let new_len = old_len + total_bins;
            self.buffer.resize(new_len, T::default());
            debug_assert!(new_len < u32::max_value() as usize);
            self.ranges.insert(node_id, (old_len as u32, new_len as u32));
        }
    }

    pub fn free_histograms(&mut self, node_id: usize) {
        self.recycle_list.push(node_id);
    }
}






#[cfg(test)]
mod test {
    use super::{HistStore};

    #[test]
    fn histograms() {
        let cardinalities = vec![4, 6, 16];
        let mut store = HistStore::<f32>::new(cardinalities.iter().cloned());
        let ptr0;
        let ptr1;
        let ptr2;

        {
            store.alloc_histograms(0);

            assert_eq!(store.get_histogram(0, 0).len(), 4);
            assert_eq!(store.get_histogram(0, 1).len(), 6);
            assert_eq!(store.get_histogram(0, 2).len(), 16);

            ptr0 = &store.get_histogram(0, 0)[0] as *const f32;
        }

        {
            store.alloc_histograms(1);
            store.free_histograms(0);
            store.alloc_histograms(2);

            ptr1 = &store.get_histogram(1, 0)[0] as *const f32;
        }

        {
            ptr2 = &store.get_histogram(2, 0)[0] as *const f32;

            assert_ne!(ptr0, ptr1);
            assert_eq!(ptr0, ptr2); // histograms reused
        }
    }
}
