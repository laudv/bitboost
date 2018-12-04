use bits::FullBitSet;

pub enum FeatureFormat {
    /// Representation for raw integer data
    Ints(Vec<i64>),

    BitSets(Vec<FullBitSet>),
}

pub struct Feature {
    name: String,
    index: usize,
    representations: Vec<FeatureFormat>,
}

pub struct Dataset {
    features: Vec<Feature>
}
