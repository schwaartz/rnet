use ndarray::{Array1, Array2};
use rand::{rngs::StdRng, Rng, SeedableRng};
use crate::{dataset::Dataset, loss::Loss, models::{feedforward::hidden_layer::HiddenLayer, InputLayer, OutputLayer}};
use chrono::Utc;

const RANDSEED: u64 = 0;
const DEFAULT_LEARNING_RATE: f32 = 0.1;
const DEFAULT_BATCH_SIZE: usize = 32;
const DEFAULT_EPOCHS: usize = 10;

/// A representation of a feedforward neural network
pub struct FeedForward {
    input_layer: InputLayer,
    hidden_layers: Vec<HiddenLayer>,
    output_layer: OutputLayer,
    loss: Box<dyn Loss>,
    
    // Learning parameters
    learning_rate: f32,
    batch_size: usize,
    epochs: usize,
    verbose: bool,
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
            verbose: false,
        };
        instance.initialize_weights_and_biases();
        instance
    }

    /// Sets the learning rate for training
    pub fn set_learning_rate(&mut self, learning_rate: f32) {
        self.learning_rate = learning_rate;
    }

    /// Sets the batch size for training
    pub fn set_batch_size(&mut self, batch_size: usize) {
        self.batch_size = batch_size;
    }

    /// Sets the number of epochs for training
    pub fn set_epochs(&mut self, epochs: usize) {
        self.epochs = epochs;
    }

    /// Sets the verbosity of the training process
    pub fn set_verbose(&mut self, verbose: bool) {
        self.verbose = verbose;
    }

    /// Predicts the output for a given input
    pub fn predict(&self, input: &Array1<f32>) -> Array1<f32> {
        let mut current_output = self.input_layer.compute_output(input);
        for layer in &self.hidden_layers {
            current_output = layer.compute_output(&current_output);
        }
        self.output_layer.compute_output(&current_output)
    }

    /// Trains the feedforward network using the provided dataset and the
    /// backpropagation algorithm
    pub fn train(&mut self, dataset: &Dataset) {
        for epoch in 1..=self.epochs {
            self.log(format!("Epoch {}/{} started", epoch, self.epochs));
            let (mut processed, total) = (0, dataset.len());
            for batch in dataset.random_iterator(self.batch_size) {
                self.print_progress_bar(processed, total);
                processed += batch.len();
                for (input, output) in batch {
                    self.backprop(input, output); // Could use parallelisation
                }
            }
        }
    }

    /// Performs the backpropagation algorithm on a single input/output pair
    fn backprop(&mut self, input: &Array1<f32>, output: &Array1<f32>) {
        assert!(input.len() == output.len(), "Input length ({}) must match output length ({})", input.len(), output.len());
        let gradient = self.calculate_gradient(input, output);
        unimplemented!()
    }

    fn calculate_gradient(&self, input: &Array1<f32>, output: &Array1<f32>) -> Vec<(Array1<f32>, Array2<f32>)> {
        unimplemented!()
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

    /// Prints a progress bar to the console of a fixed length if verbosity is enabled
    fn print_progress_bar(&self, processed: usize, total: usize) {
        if !self.verbose {
            return;
        }
        let bar_length = 30;
        let progress = (processed as f32 / total as f32 * bar_length as f32).round() as usize;
        let bar: String = "=".repeat(progress) + ">" + &" ".repeat(bar_length - progress - 1);
        print!("\r[{}] {}/{}", bar, processed, total);
    }

    /// Logs a message if verbosity is enabled and it adds a timestamp
    fn log(&self, message: String) {
        if self.verbose {
            let timestamp = Utc::now().format("%H:%M:%S");
            println!("[{}] {}", timestamp, message);
        }
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

    /// Tests the forward pass of the network
    #[test]
    fn test_predict() {
        let model = FeedForward::new(
            InputLayer::new(3),
            Vec::<HiddenLayer>::new(),
            OutputLayer::new(2, Box::new(Linear)),
            Box::new(CrossEntropy),
        );

        let input = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let output = model.predict(&input);

        let w = model.output_layer.weights.as_ref().unwrap();
        let b = model.output_layer.biases.as_ref().unwrap();
        let expected = w.dot(&input) + b;

        assert_eq!(output, expected);
    }
}