use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::error::Error;

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

/// Represents the use case for an RNet neural network
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UseCase {
    /// A use case for classification tasks. This will use cross-entropy loss and softmax output activation
    Classification,
    /// A use case for regression tasks. This will use mean squared error loss and no output activation
    Regression,
    /// A default use case. This will use mean squared error loss and no output activation
    Default,
}

/// Represents an RNet neural network instance.
/// It encapsulates the network architecture, training parameters, and other
/// relevant information for training, testing and general usage.
#[derive(Debug, Clone)]
pub struct RNet {
    pub network: Network,
    pub learning_rate: f64,
    pub batch_size: usize,
    pub epochs: usize,
    pub output_activation: OutputActivation,
    pub use_case: UseCase,
}

impl RNet {
    /// Generates a new RNet instance with the given architecture. The loss function
    /// and the output activation functions are chosen based on the use case.
    pub fn new(shape: Vec<usize>, activations: Vec<Activation>, use_case: UseCase) -> Self {
        assert!(shape.len() >= 2);
        assert!(shape.len() == activations.len());
        for dim in shape.iter() {
            assert!(*dim > 0);
        }

        // Define all the layers, except for the input layer
        let mut layers = Vec::new();
        for i in 1..shape.len() {
            layers.push(Layer::new(
                shape[i],
                activations[i],
                Self::rand_bias(shape[i]),
                Self::rand_weights(shape[i], shape[i - 1]),
            ));
        }

        let loss = match use_case {
            UseCase::Classification => Loss::CrossEntropy,
            UseCase::Regression => Loss::MSE,
            UseCase::Default => Loss::MSE,
        };

        let output_activation = match use_case {
            UseCase::Classification => OutputActivation::Softmax,
            UseCase::Regression => OutputActivation::None,
            UseCase::Default => OutputActivation::None,
        };

        RNet {
            network: Network::new(shape[0], layers, loss, *activations.first().unwrap()),
            learning_rate: DEFAULT_LEARNING_RATE,
            batch_size: DEFAULT_BATCH_SIZE,
            epochs: DEFAULT_EPOCHS,
            output_activation,
            use_case,
        }
    }

    /// Trains the neural network on the provided training data
    pub fn train(&mut self, data: &Dataset) {
        self.network.train(data, self.batch_size, self.epochs, self.learning_rate);
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

    /// Computes the mean squared errors between the network's predictions and the target outputs
    /// and returns the average error.
    pub fn mean_squared_error(&self, dataset: &Dataset) -> f64 {
        let mut squared_errors = vec![0.0; dataset.len()];
        for (input, target) in dataset.iterator() {
            let test_output = self.network.forward_prop(&input);
            squared_errors.push(
                (test_output - target).mapv(|x| x.powi(2) / target.len() as f64).sum()
            );
        }
        squared_errors.iter().sum::<f64>() / squared_errors.len() as f64
    }

    /// Calculates the accuracy of the network.
    /// This function assumes that the network is used for classification.
    /// Calling this function on a non-classifier network.
    pub fn accuracy(&self, dataset: &Dataset) -> f64 {
        let get_max = |arr: &Array1<f64>| {
            arr.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(idx, _)| idx)
                .unwrap()
        };
        let mut correct_predictions = 0;
        let mut total_predictions = 0;
        for (input, target) in dataset.iterator() {
            let test_output = self.network.forward_prop(&input);
            let predicted_class = get_max(&test_output);
            let actual_class = get_max(&target);
            if predicted_class == actual_class {
                correct_predictions += 1;
            }
            total_predictions += 1;
        }
        correct_predictions as f64 / total_predictions as f64
    }

    /// Saves the RNnet weights to a file
    pub fn save_weights(&self, file_path: &str) -> Result<(), Box<dyn Error>> {
        let file = File::create(file_path)?;
        let mut writer = BufWriter::new(file);
        for (i, layer) in self.network.layers.iter().enumerate() {
            writeln!(writer, "\n Layer {}, Activation {:?}", i, layer.activation)?;
            writeln!(writer, "Biases:")?;
            for b in layer.bias.iter() {
                writeln!(writer, "B {}", b)?;
            }
            writeln!(writer, "Weights:")?;
            for row in 0..layer.weights.nrows() {
                let row_str = layer.weights.row(row).iter()
                    .map(|w| w.to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                writeln!(writer, "W {}", row_str)?;
            }
        };
        Ok(())
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

