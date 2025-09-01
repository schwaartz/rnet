use ndarray::{Array1, Array2};
use rand::{rngs::StdRng, Rng, SeedableRng};
use crate::models::feedforward::hidden_layer::HiddenLayer;

const RANDSEED: u64 = 0;

struct FeedForward {
    hidden_layers: Vec<HiddenLayer>,
}

impl FeedForward {
    /// Generates a random bias vector
    fn rand_bias(size: usize) -> Array1<f32> {
        let mut rng = StdRng::seed_from_u64(RANDSEED);
        Array1::from_shape_fn(size, |_| rng.random_range(-1.0..1.0))
    }

    /// Generates a random weight matrix
    fn rand_weights(rows: usize, cols: usize) -> Array2<f32> {
        let mut rng = StdRng::seed_from_u64(RANDSEED);
        Array2::from_shape_fn((rows, cols), |_| rng.random_range(-1.0..1.0))
    }
}