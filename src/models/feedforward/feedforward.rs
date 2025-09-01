use ndarray::{Array1, Array2};
use rand::{rngs::StdRng, Rng, SeedableRng};
use crate::{loss::Loss, models::{feedforward::hidden_layer::HiddenLayer, InputLayer, OutputLayer}};

const RANDSEED: u64 = 0;
const DEFAULT_LEARNING_RATE: f32 = 0.1;
const DEFAULT_BATCH_SIZE: usize = 32;
const DEFAULT_EPOCHS: usize = 10;

/// A representation of a feedforward neural network
struct FeedForward {
    input_layer: InputLayer,
    hidden_layers: Vec<HiddenLayer>,
    output_layer: OutputLayer,
    loss: Box<dyn Loss>,
    
    // Learning parameters
    learning_rate: f32,
    batch_size: usize,
    epochs: usize,
}

impl FeedForward {
    /// Creates a new FeedForward model with the given layers and loss function
    pub fn new(
        input_layer: InputLayer,
        hidden_layers: Vec<HiddenLayer>,
        output_layer: OutputLayer,
        loss: Box<dyn Loss>,
    ) -> Self {
        let mut instance = Self {
            input_layer,
            hidden_layers,
            output_layer,
            loss,
            learning_rate: DEFAULT_LEARNING_RATE,
            batch_size: DEFAULT_BATCH_SIZE,
            epochs: DEFAULT_EPOCHS,
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
        self.output_layer.biases = Some(Self::rand_bias(self.output_layer.dim));
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

#[cfg(test)]
mod tests {
    use crate::{activation::{Linear}, loss::CrossEntropy};
    use super::*;

    #[test]
    fn test_feedforward_initialization() {
        let input_layer = InputLayer::new(3);
        let hidden_layers = vec![
            HiddenLayer::new(4, Box::new(Linear)),
            HiddenLayer::new(5, Box::new(Linear)),
        ];
        let output_layer = OutputLayer::new(2, Box::new(Linear));
        let loss = Box::new(CrossEntropy);

        let model = FeedForward::new(
            input_layer,
            hidden_layers,
            output_layer,
            loss,
        );

        // Biases
        assert_eq!(model.hidden_layers[0].biases.as_ref().unwrap().len(), 4);
        assert_eq!(model.hidden_layers[1].biases.as_ref().unwrap().len(), 5);
        assert_eq!(model.output_layer.biases.as_ref().unwrap().len(), 2);
        
        // Weights
        assert_eq!(model.hidden_layers[0].weights.as_ref().unwrap().shape(), &[4, 3]);
        assert_eq!(model.hidden_layers[1].weights.as_ref().unwrap().shape(), &[5, 4]);
        assert_eq!(model.output_layer.weights.as_ref().unwrap().shape(), &[2, 5]);
    }
}