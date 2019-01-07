use fnv::FnvHashMap as HashMap;
use dataset::Dataset;

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
            if let Some((card, _)) = f.get_cat_repr() { card as u32 }
            else { panic!("non categorical feature not supported"); }
        }))
    }

    fn get_histograms_range(&self, node_id: usize) -> (usize, usize) {
        let (lo, hi) = self.ranges[&node_id];
        (lo as usize, hi as usize)
    }

    pub fn has_histograms_for_node(&self, node_id: usize) -> bool {
        self.ranges.contains_key(&node_id)
    }

    pub fn histograms_for_node<'a>(&'a mut self, node_id: usize) -> Histograms<'a, T> {
        let (lo, hi) = self.get_histograms_range(node_id);
        Histograms {
            buffer: &mut self.buffer[lo..hi],
            hist_layout: &self.hist_layout,
        }
    }

    pub fn alloc_histograms(&mut self, node_id: usize)
    where T: Default + Clone {
        debug_assert!(!self.has_histograms_for_node(node_id));
        println!("{:?}", self.recycle_list);
        if let Some(freed_node_id) = self.recycle_list.pop() {
            // reuse previous histograms
            println!("hello world! {} -> {}", freed_node_id, node_id);
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

pub struct Histograms<'a, T: 'a> {
    buffer: &'a mut [T],
    hist_layout: &'a [u32],
}

impl <'a, T: 'a> Histograms<'a, T> {

    fn get_range(&self, feat_id: usize) -> (usize, usize) {
        let lo = self.hist_layout[feat_id];
        let hi = self.hist_layout[feat_id + 1];
        (lo as usize, hi as usize)
    }

    /// Get the histogram for the feature in the given node.
    pub fn get_hist(&self, feat_id: usize) -> &[T] {
        let (lo, hi) = self.get_range(feat_id);
        &self.buffer[lo..hi]
    }

    /// Get the mutable histogram for the feature in the given node.
    pub fn get_hist_mut(&mut self, feat_id: usize) -> &mut [T] {
        let (lo, hi) = self.get_range(feat_id);
        &mut self.buffer[lo..hi]
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
            let hs0 = store.histograms_for_node(0);

            assert_eq!(hs0.get_hist(0).len(), 4);
            assert_eq!(hs0.get_hist(1).len(), 6);
            assert_eq!(hs0.get_hist(2).len(), 16);

            ptr0 = &hs0.get_hist(0)[0] as *const f32;
        }

        {
            store.alloc_histograms(1);
            store.free_histograms(0);
            store.alloc_histograms(2);

            let hs1 = store.histograms_for_node(1);
            ptr1 = &hs1.get_hist(0)[0] as *const f32;
        }

        {
            let hs2 = store.histograms_for_node(2);
            ptr2 = &hs2.get_hist(0)[0] as *const f32;

            assert_ne!(ptr0, ptr1);
            assert_eq!(ptr0, ptr2); // histograms reused
        }
    }
}
