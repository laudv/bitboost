
pub enum FeatureFormat {
    /// Representation for raw integer data
    Ints(Vec<i32>),

    BitSets(Vec<i64>),
}

pub struct Feature {
    name: String,
    index: usize,
    representations: Vec<FeatureFormat>,
}

pub struct Dataset {
    features: Vec<Feature>
}
