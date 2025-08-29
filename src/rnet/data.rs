use ndarray::Array1;
use rand::seq::SliceRandom;
use rand::{SeedableRng, rngs::StdRng};

const RANDSEED: u64 = 0;

/// A struct that represents a dataset for training a neural network
#[derive(Debug, Clone)]
pub struct Dataset {
    pub inputs: Vec<Array1<f64>>,
    pub targets: Vec<Array1<f64>>,
    randseed: u64,
}

impl Dataset {
    /// Creates a new Dataset
    pub fn new(inputs: Vec<Array1<f64>>, targets: Vec<Array1<f64>>) -> Self {
        assert!(inputs.len() == targets.len(), "Inputs len {} != Targets len {}", inputs.len(), targets.len());
        Dataset { inputs, targets, randseed: RANDSEED }
    }

    /// Returns a random iterator over the dataset with (input, target) pairs
    pub fn random_iterator(&self) -> impl Iterator<Item = (&Array1<f64>, &Array1<f64>)> {
        let mut rng = StdRng::seed_from_u64(self.randseed);
        let mut indices: Vec<usize> = (0..self.inputs.len()).collect();
        indices.shuffle(&mut rng);
        indices.into_iter().map(move |i| (&self.inputs[i], &self.targets[i]))
    }

    /// Returns an iterator over the dataset with (input, target) pairs
    pub fn iterator(&self) -> impl Iterator<Item = (&Array1<f64>, &Array1<f64>)> {
        self.inputs.iter().zip(&self.targets)
    }

    /// Returns the number of samples in the dataset
    pub fn len(&self) -> usize {
        self.inputs.len()
    }

    /// Sets the random seed for the dataset
    pub fn set_randseed(&mut self, seed: u64) {
        self.randseed = seed;
    }
}