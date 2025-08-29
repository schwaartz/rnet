use std::mem::uninitialized;

use crate::rnet::activation::Activation;
use crate::rnet::activation::OutputActivation;
use crate::rnet::network::Network;
use crate::rnet::layer::Layer;
use crate::rnet::loss::Loss;
use crate::rnet::data::Dataset;

use ndarray::Array1;
use ndarray::Array2;
use rand::{rngs::StdRng, Rng, SeedableRng};

const RANDSEED: u64 = 0;
const DEFAULT_LEARNING_RATE: f64 = 0.001;
const DEFAULT_BATCH_SIZE: usize = 32;
const DEFAULT_EPOCHS: usize = 10;

/// Represents an RNet neural network instance.
/// It encapsulates the network architecture, training parameters, and other
/// relevant information for training, testing and general usage.
pub struct RNet {
    network: Network,
    learning_rate: f64,
    batch_size: usize,
    epochs: usize,
    output_activation: OutputActivation,
}

impl RNet {
    /// Generates a new RNet instance with a default neural network architecture.
    /// The default architecture has a shape given by the `shape` argument, activation
    /// functions given by the `activations` argument and it has a mean squared error loss function.
    pub fn new_default_nn(shape: Vec<usize>, activations: Vec<Activation>) -> Self {
        assert!(shape.len() >= 2);
        assert!(shape.len() == activations.len() + 1);
        for dim in shape.iter() {
            assert!(*dim > 0);
        }

        // Define all the layers, except for the input layer
        let mut layers = Vec::new();
        for i in 1..shape.len() {
            layers.push(Layer::new(
                shape[i],
                activations[i - 1],
                Self::rand_bias(shape[i]),
                Self::rand_weights(shape[i - 1], shape[i]),
            ));
        }

        RNet {
            network: Network::new(shape[0], layers, Loss::MSE),
            learning_rate: DEFAULT_LEARNING_RATE,
            batch_size: DEFAULT_BATCH_SIZE,
            epochs: DEFAULT_EPOCHS,
            output_activation: OutputActivation::None,
        }
    }

    /// Generates a new RNet instance with a default classifier neural network architecture.
    /// The default architecture has a shape specified by the `shape` argument, activation
    /// functions given by the `activations` argument and it has a cross-entropy loss function.
    /// The size of the activations vector is 2 smaller than the shape, since the first layer
    /// has no activation and the last layer uses softmax.
    pub fn new_classifier_nn(shape: Vec<usize>, activations: Vec<Activation>) -> Self {
        assert!(shape.len() >= 2);
        assert!(shape.len() == activations.len() + 2);
        for dim in shape.iter() {
            assert!(*dim > 0);
        }

        // Define all the layers, except for the input layer
        let mut layers = Vec::new();
        for i in 1..shape.len() {
            layers.push(Layer::new(
                shape[i],
                activations[i - 1],
                Self::rand_bias(shape[i]),
                Self::rand_weights(shape[i - 1], shape[i]),
            ));
        }

        RNet {
            network: Network::new(shape[0], layers, Loss::CrossEntropy),
            learning_rate: DEFAULT_LEARNING_RATE,
            batch_size: DEFAULT_BATCH_SIZE,
            epochs: DEFAULT_EPOCHS,
            output_activation: OutputActivation::Softmax,
        }
    }

    /// Trains the neural network on the provided training data
    pub fn train(&mut self, data: Dataset) {
        self.network.train(data, self.batch_size, self.epochs, self.learning_rate);
    }

    /// Evaluates the neural network on the provided test data and returns an evaluation object
    pub fn evaluate(&self, data: Dataset) -> f64 {
        unimplemented!();
    }

    /// Makes a prediction using the trained neural network
    pub fn predict(&self, input: &Array1<f64>) -> Array1<f64> {
        let logits = self.network.forward_prop(input);
        self.output_activation.func(logits) // e.g. apply softmax
    }

    /// Sets the learning rate for training (otherwise default)
    pub fn set_learning_rate(&mut self, learning_rate: f64) {
        self.learning_rate = learning_rate;
    }

    /// Sets the batch size for training (otherwise default)
    pub fn set_batch_size(&mut self, batch_size: usize) {
        self.batch_size = batch_size;
    }

    /// Sets the number of training epochs (otherwise default)
    pub fn set_epochs(&mut self, epochs: usize) {
        self.epochs = epochs;
    }

    /// Generates a random bias vector
    fn rand_bias(size: usize) -> Array1<f64> {
        let mut rng = StdRng::seed_from_u64(RANDSEED);
        Array1::from_shape_fn(size, |_| rng.random_range(-1.0..1.0))
    }

    /// Generates a random weight matrix
    fn rand_weights(rows: usize, cols: usize) -> Array2<f64> {
        let mut rng = StdRng::seed_from_u64(RANDSEED);
        Array2::from_shape_fn((rows, cols), |_| rng.random_range(-1.0..1.0))
    }
}