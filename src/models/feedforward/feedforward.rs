use ndarray::{Array1, Array2};
use rand::{rngs::StdRng, Rng, SeedableRng};
use crate::{dataset::Dataset, loss::Loss, models::{feedforward::hidden_layer::HiddenLayer, InputLayer, OutputLayer}};
use chrono::Utc;

const RANDSEED: u64 = 0;
const DEFAULT_LEARNING_RATE: f32 = 0.1;
const DEFAULT_BATCH_SIZE: usize = 32;
const DEFAULT_EPOCHS: usize = 10;
const GRADIENT_CLIP: f32 = 5.0;

/// A representation of a feedforward neural network
pub struct FeedForward {
    pub input_layer: InputLayer,
    pub hidden_layers: Vec<HiddenLayer>,
    pub output_layer: OutputLayer,
    pub loss: Box<dyn Loss>,
    pub randseed: u64,

    // Learning parameters
    pub gradient_clip: f32,
    pub learning_rate: f32,
    pub batch_size: usize,
    pub epochs: usize,
    pub verbose: bool,
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
            randseed: RANDSEED,
            gradient_clip: GRADIENT_CLIP,
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

    /// Sets the random seed for weight initialization
    pub fn set_randseed(&mut self, rand_seed: u64) {
        self.randseed = rand_seed;
    }

    /// Sets the verbosity of the training process
    pub fn set_verbose(&mut self, verbose: bool) {
        self.verbose = verbose;
    }

    /// Sets the gradient clipping value
    pub fn set_gradient_clip(&mut self, gradient_clip: f32) {
        self.gradient_clip = gradient_clip;
    }

    /// Predicts the output for a given input
    pub fn predict(&self, input: &Array1<f32>) -> Array1<f32> {
        let mut current_output = self.input_layer.compute_output(input);
        for layer in &self.hidden_layers {
            current_output = layer.compute_output(&current_output);
        }
        self.output_layer.compute_output(&current_output)
    }

    /// Calculates the accuracy of the model on the given dataset.
    /// This metric is most useful for classifiers.
    pub fn accuracy(&self, dataset: &Dataset) -> f32 {
        assert!(!dataset.inputs.is_empty(), "Dataset must not be empty to calculate accuracy");
        let get_max = |arr: &Array1<f32>| {
            let (mut max_i, mut max_val) = (-1, f32::MIN);
            for (i, val) in arr.iter().enumerate() {
                if val > &max_val {
                    (max_i, max_val) = (i as i32, *val)
                }
            }
            max_i
        };
        let mut correct = 0;
        for (input, target) in dataset.iter() {
            let output = self.predict(input);
            let predicted_class = get_max(&output);
            let actual_class = get_max(target);
            if predicted_class == actual_class {
                correct += 1;
            }
        }
        correct as f32 / dataset.len() as f32
    }

    /// Calculates the mean squared error of the model on the given dataset
    pub fn mse(&self, dataset: &Dataset) -> f32 {
        assert!(!dataset.inputs.is_empty(), "Dataset must not be empty to calculate MSE");
        let mut total_error = 0.0;
        for (input, target) in dataset.iter() {
            let output = self.predict(input);
            total_error += (&output - target).pow2().sum();
        }
        total_error / dataset.len() as f32
    }

    /// Trains the feedforward network using the provided dataset and the
    /// backpropagation algorithm
    pub fn train(&mut self, dataset: &Dataset) {
        for epoch in 1..=self.epochs {
            self.log(format!("Epoch {}/{} started", epoch, self.epochs));
            let (mut processed, total) = (0, dataset.len());
            for batch in dataset.rand_iter(self.batch_size) {
                self.print_progress_bar(processed, total);
                processed += batch.len();
                for (input, output) in batch {
                    self.backprop(input, output); // Could use parallelisation
                }
            }
            if self.verbose { println!(""); } // New line after progress bar is finished
        }
    }

    /// Performs the backpropagation algorithm on a single input/output pair
    pub fn backprop(&mut self, input: &Array1<f32>, target: &Array1<f32>) {
        assert_eq!(input.len(), self.input_layer.dim, "Input length must match input layer dimension");
        assert_eq!(target.len(), self.output_layer.dim, "Target length must match output layer dimension");
        
        let gradient = self.calculate_gradient(input, target);
        assert_eq!(gradient.len(), self.hidden_layers.len() + 1, "Gradient length must match number of hidden layers + 1");

        // Update all the weights and biases in the hidden layers and the output layer
        for l in 0..self.hidden_layers.len() {
            let (weight_grad, bias_grad) = gradient[l].clone();
            let old_w = self.hidden_layers[l].weights.as_ref().unwrap();
            let old_b = self.hidden_layers[l].biases.as_ref().unwrap();
            let new_w = old_w - weight_grad * self.learning_rate;
            let new_b = old_b - bias_grad * self.learning_rate;
            self.hidden_layers[l].weights = Some(new_w);
            self.hidden_layers[l].biases = Some(new_b);
        }
        
        // Update the output layer
        let (weight_grad, bias_grad) = gradient.last().unwrap().clone();
        let old_w = self.output_layer.weights.as_ref().unwrap();
        let old_b = self.output_layer.biases.as_ref().unwrap();
        let new_w = old_w - weight_grad * self.learning_rate;
        let new_b = old_b - bias_grad * self.learning_rate;
        self.output_layer.weights = Some(new_w);
        self.output_layer.biases = Some(new_b);
    }

    /// Calculates the gradient for each of the layers of the network (excluding
    /// the input layer, because it has not weights or biases) in the format
    /// Vec<(Array2<f32>, Array1<f32>)> where every first entry of the tuple are
    /// the partial derivatives of the weights and the second entry of the tuple
    /// are the partial derivatives of the biases.
    fn calculate_gradient(&self, input: &Array1<f32>, target: &Array1<f32>) -> Vec<(Array2<f32>, Array1<f32>)> {
        // We calculate the output and store all the intermediate results before
        // applying the activation functions.
        let (a, z) = self.calculate_a_and_z(input);

        // The variable we will be returning
        let mut gradient = Vec::<(Array2<f32>, Array1<f32>)>::new();

        // Now we compute $\partial C / \partial a^{(L)}$
        // with C the cost/loss function and L the final layer
        let output = a.last().unwrap();
        let mut grad_a: Array1<f32> = self.loss.compute_gradient(output, target);

        // Now we loop through each of the layers backwards, excluding the input layer
        for l in (1..self.hidden_layers.len() + 2).rev() { // Includes the output layer
            // Calculation of the gradient for activations $\partial C / \partial a^{(l)}$,
            // the gradient for the weights $\partial C / \partial w^{(l)}$ and the gradient
            // of the biases: $\partial C / \partial b^{(l)}$
            let grad_w_l = self.calculate_weight_gradient(l, &a, &z, &grad_a);
            let grad_b_l = self.calculate_bias_gradient(l, &z, &grad_a);
            if l > 1 {
                grad_a = self.calculate_a_gradient(l - 1, &z, &grad_a);
            }
            gradient.push((grad_w_l, grad_b_l));
        }

        gradient.reverse();
        self.clip_gradient(&mut gradient);
        gradient
    }

    /// Private method that clips the gradient when it goes out of bounds.
    /// This prevents the exploding gradient problem.
    fn clip_gradient(&self, gradient: &mut Vec<(Array2<f32>, Array1<f32>)>) {
        let mut norm: f32 = 0.0;
        for (grad_w, grad_b) in gradient.iter() {
            norm += grad_w.mapv(|x| x.powi(2)).sum();
            norm += grad_b.mapv(|x| x.powi(2)).sum();
        }
        norm = norm.sqrt();
        if norm > self.gradient_clip {
            let scale = self.gradient_clip / norm;
            for (grad_w, grad_b) in gradient.iter_mut() {
                grad_w.mapv_inplace(|x| x * scale);
                grad_b.mapv_inplace(|x| x * scale);
            }
        }
    }

    /// Private function to help calculate the a and z vectors for every layer.
    /// So this function performs the forward step of the backpropagation algorithm.
    fn calculate_a_and_z(&self, input: &Array1<f32>) -> (Vec<Array1<f32>>, Vec<Array1<f32>>) {
        let mut z = Vec::<Array1<f32>>::new();
        let mut a = Vec::<Array1<f32>>::new();

        // Input layer
        z.push(input.clone());
        let mut curr_output = self.input_layer.compute_output(input);
        a.push(curr_output.clone());

        // Hidden layers
        for layer in &self.hidden_layers {
            let z_l = layer.compute_logits(&curr_output);
            curr_output = layer.activation.compute_value(&z_l);
            a.push(curr_output.clone());
            z.push(z_l);
        }

        // Output layer
        let z_output_layer = self.output_layer.compute_logits(&curr_output);
        a.push(self.output_layer.activation.compute_value(&z_output_layer));
        z.push(z_output_layer);
        (a, z)
    } 

    /// Private function to help calculate the weight gradient for a given layer.
    /// This function computes $\partial C / \partial w^{(l)}$.
    fn calculate_weight_gradient(
        &self,
        l: usize,
        a: &Vec<Array1<f32>>,
        z: &Vec<Array1<f32>>,
        grad_a: &Array1<f32>,
    ) -> Array2<f32> {
        let layer = if l < self.hidden_layers.len() + 1 {
            &self.hidden_layers[l - 1]
        } else {
            &self.output_layer
        };
        let activation_deriv = &layer.activation.compute_derivative(&z[l]);

        // $(\sigma'(z^{(l)}) \odot \partial C / \partial a^{(l)}) \cdot a^{(l-1) T}$
        let col_vec = (activation_deriv * grad_a).insert_axis(ndarray::Axis(1));
        let row_vec = a[l - 1].clone().insert_axis(ndarray::Axis(0));
        col_vec.dot(&row_vec)
    }

    /// Private function to help calculate the bias gradient for a given layer.
    /// This function computes $\partial C / \partial b^{(l)}$.
    fn calculate_bias_gradient(&self, l: usize, z: &Vec<Array1<f32>>, grad_a: &Array1<f32>) -> Array1<f32> {
        let layer = if l < self.hidden_layers.len() + 1 {
            &self.hidden_layers[l - 1]
        } else {
            &self.output_layer
        };

        // $\partial C / \partial b^{(l)} = \sigma'(z^{(l)}) \odot \partial C / \partial a^{(l)}$
        grad_a * &layer.activation.compute_derivative(&z[l])
    }

    /// Private function to help calculate the new a gradient for a given layer.
    /// This function computes $\partial C / \partial a^{(l)}$ using the chain rule.
    fn calculate_a_gradient(&self, l: usize, z: &Vec<Array1<f32>>, grad_a: &Array1<f32>) -> Array1<f32> {
        let next_layer = if l >= self.hidden_layers.len() {
            &self.output_layer // layer l + 1
        } else {
            &self.hidden_layers[l] // layer l + 1
        };

        // $\partial C / \partial a^{(l)} = (W^{(l+1)})^T \cdot (\sigma'(z^{(l+1)}) \odot \partial C / \partial a^{(l+1)})$
        let weights = next_layer.weights.as_ref().unwrap();
        let weights_t = weights.clone().reversed_axes(); // Transposes the array
        let activation_deriv = &next_layer.activation.compute_derivative(&z[l + 1]);
        weights_t.dot(&(activation_deriv * grad_a))   
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
            self.hidden_layers[l].weights = Some(self.rand_weights(rows, cols));
            self.hidden_layers[l].biases = Some(self.rand_bias(rows));
        }

        // Initialize the output layer
        self.output_layer.biases = Some(self.rand_bias(self.output_layer.dim));
        if self.hidden_layers.is_empty() {
            self.output_layer.weights = Some(self.rand_weights(
                self.output_layer.dim,
                self.input_layer.dim,
            ));
        } else {
            self.output_layer.weights = Some(self.rand_weights(
                self.output_layer.dim,
                self.hidden_layers.last().unwrap().dim,
            ));
        }
    }

    /// Generates a random bias
    fn rand_bias(&self, size: usize) -> Array1<f32> {
        let mut rng = StdRng::seed_from_u64(self.randseed);
        Array1::from_shape_fn(size, |_| rng.random_range(-0.1..0.1))
    }

    /// Generates a random weight matrix
    fn rand_weights(&self, rows: usize, cols: usize) -> Array2<f32> {
        let mut rng = StdRng::seed_from_u64(self.randseed);
        Array2::from_shape_fn((rows, cols), |_| rng.random_range(-0.1..0.1))
    }

    /// Prints a progress bar to the console of a fixed length if verbosity is enabled
    fn print_progress_bar(&self, processed: usize, total: usize) {
        if !self.verbose {
            return;
        }
        let bar_length = 30;
        let progress = (processed as f32 / total as f32 * bar_length as f32).round() as usize;
        let bar: String = "=".repeat(progress) + &" ".repeat(bar_length - progress);
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
    use crate::{activation::{Linear, Softmax}, loss::CrossEntropy};
    use crate::dataset::Dataset;
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

    #[test]
    fn test_backward_pass() {
        let mut model = FeedForward::new(
            InputLayer::new(3),
            vec![HiddenLayer::new(4, Box::new(Linear))],
            OutputLayer::new(2, Box::new(Softmax)),
            Box::new(CrossEntropy),
        );
        model.set_randseed(0);
        model.set_learning_rate(0.1);

        let input = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let target = Array1::from_vec(vec![0.0, 1.0]);
        let old_loss = model.loss.compute_value(&model.predict(&input), &target);
        for _ in 0..10 {
            model.backprop(&input, &target);
        }
        let new_loss = model.loss.compute_value(&model.predict(&input), &target);

        assert!(old_loss > new_loss);
    }

    #[test]
    fn test_train() {
        let mut model = FeedForward::new(
            InputLayer::new(3),
            vec![HiddenLayer::new(4, Box::new(Linear))],
            OutputLayer::new(2, Box::new(Softmax)),
            Box::new(CrossEntropy),
        );
        model.set_randseed(0);
        model.set_learning_rate(0.1);
        model.set_epochs(10);

        let input = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let target = Array1::from_vec(vec![0.0, 1.0]);
        let old_loss = model.loss.compute_value(&model.predict(&input), &target);
        let dataset = Dataset::new(vec![input.clone()], vec![target.clone()]);
        model.train(&dataset);
        let new_loss = model.loss.compute_value(&model.predict(&input), &target);

        assert!(old_loss > new_loss);
    }

    #[test]
    fn test_accuracy() {
        let model = FeedForward::new(
            InputLayer::new(3),
            vec![HiddenLayer::new(4, Box::new(Linear))],
            OutputLayer::new(2, Box::new(Softmax)),
            Box::new(CrossEntropy),
        );

        let input1 = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let input2 = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let target1 = Array1::from_vec(vec![0.0, 1.0]);
        let target2 = Array1::from_vec(vec![1.0, 0.0]);
        let dataset = Dataset::new(vec![input1.clone(), input2.clone()], vec![target1.clone(), target2.clone()]);
        let accuracy = model.accuracy(&dataset);
        assert_eq!(accuracy, 0.5);
    }

    #[test]
    fn test_mse() {
        let model = FeedForward::new(
            InputLayer::new(3),
            vec![HiddenLayer::new(4, Box::new(Linear))],
            OutputLayer::new(2, Box::new(Softmax)),
            Box::new(CrossEntropy),
        );

        let input1 = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let input2 = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let target1 = Array1::from_vec(vec![0.0, 1.0]);
        let target2 = Array1::from_vec(vec![1.0, 0.0]);
        let output1 = model.predict(&input1);
        let output2 = model.predict(&input2);
        let expected_mse = ((output1 - &target1).pow2().sum() + (output2 - &target2).pow2().sum()) / 2.0;
        let dataset = Dataset::new(vec![input1.clone(), input2.clone()], vec![target1.clone(), target2.clone()]);
        let output_mse = model.mse(&dataset);
        assert_eq!(output_mse, expected_mse);
    }

    #[test] 
    #[should_panic(expected = "Dataset must not be empty to calculate accuracy")]
    fn test_empty_accuracy() {
        let model = FeedForward::new(
            InputLayer::new(3),
            vec![HiddenLayer::new(4, Box::new(Linear))],
            OutputLayer::new(2, Box::new(Softmax)),
            Box::new(CrossEntropy),
        );
        let dataset = Dataset::new(vec![], vec![]);
        model.accuracy(&dataset);
    }

    #[test]
    #[should_panic(expected = "Dataset must not be empty to calculate MSE")]
    fn test_empty_mse() {
        let model = FeedForward::new(
            InputLayer::new(3),
            vec![HiddenLayer::new(4, Box::new(Linear))],
            OutputLayer::new(2, Box::new(Softmax)),
            Box::new(CrossEntropy),
        );
        let dataset = Dataset::new(vec![], vec![]);
        model.mse(&dataset);
    }
}