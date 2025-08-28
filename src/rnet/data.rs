use ndarray::Array1;
use rand::seq::SliceRandom;
use rand::{SeedableRng, rngs::StdRng};

const RANDSEED: u64 = 0;

/// A struct that represents a dataset for training a neural network
pub struct Dataset {
    pub inputs: Vec<Array1<f64>>,
    pub targets: Vec<Array1<f64>>,
    randseed: u64,
}

impl Dataset {
    /// Creates a new Dataset
    pub fn new(inputs: Vec<Array1<f64>>, targets: Vec<Array1<f64>>) -> Self {
        Dataset { inputs, targets, randseed: RANDSEED }
    }

    /// Returns a random iterator over the dataset
    pub fn random_iterator(&self) -> impl Iterator<Item = (&Array1<f64>, &Array1<f64>)> {
        let mut rng = StdRng::seed_from_u64(self.randseed);
        let mut indices: Vec<usize> = (0..self.inputs.len()).collect();
        indices.shuffle(&mut rng);
        indices.into_iter().map(move |i| (&self.inputs[i], &self.targets[i]))
    }

    /// Sets the random seed for the dataset
    pub fn set_randseed(&mut self, seed: u64) {
        self.randseed = seed;
    }
}