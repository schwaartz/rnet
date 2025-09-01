use ndarray::{Array1, Array2};
use rand::{rngs::StdRng, Rng, SeedableRng};
use crate::models::{feedforward::hidden_layer::HiddenLayer, InputLayer, OutputLayer};

const RANDSEED: u64 = 0;

/// A representation of a feedforward neural network
struct FeedForward {
    input_layer: InputLayer,
    hidden_layers: Vec<HiddenLayer>,
    output_layer: OutputLayer,
}

impl FeedForward {
    /// Creates a new FeedForward model with the given layer
    pub fn new(
        input_layer: InputLayer,
        hidden_layers: Vec<HiddenLayer>,
        output_layer: OutputLayer,
    ) -> Self {
        let mut instance = Self {
            input_layer,
            hidden_layers,
            output_layer,
        };
        instance.initialize_weights_and_biases();
        instance
    }

    /// Initializes the weights and biases as random matrices and vectors
    fn initialize_weights_and_biases(&mut self) {
        // Initialize the hidden layers
        for l in 0..self.hidden_layers.len() {
            let (rows, cols) = if l == 0 {
                (self.hidden_layers[l].dim, self.input_layer.dim)
            } else {
                (self.hidden_layers[l].dim, self.hidden_layers[l - 1].dim)
            };
            self.hidden_layers[l].weights = Some(Self::rand_weights(rows, cols));
            self.hidden_layers[l].biases = Some(Self::rand_bias(rows));
        }

        // Initialize the output layer
        if self.hidden_layers.is_empty() {
            self.output_layer.weights = Some(Self::rand_weights(
                self.output_layer.dim,
                self.input_layer.dim,
            ));
        } else {
            self.output_layer.weights = Some(Self::rand_weights(
                self.output_layer.dim,
                self.hidden_layers.last().unwrap().dim,
            ));
        }
    }

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