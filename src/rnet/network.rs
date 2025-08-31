use ndarray::{Array1, Array2};
use crate::rnet::layer::Layer;
use crate::rnet::loss::Loss;
use crate::rnet::data::Dataset;

/// The Network struct implements a simple neural network
#[derive(Debug, Clone)]
pub struct Network {
    pub input_dim: usize,
    pub layers: Vec<Layer>,
    pub loss: Loss,
}

impl Network {
    /// Creates a new Network
    pub fn new(input_dim: usize, layers: Vec<Layer>, loss: Loss) -> Self {
        Network { input_dim, layers, loss }
    }

    /// Calculates the output of the network given an input vector.
    /// It also checks if the input size matches the network's input dimension.
    pub fn forward_prop(&self, input: &Array1<f64>) -> Array1<f64> {
        assert!(input.len() == self.input_dim);

        let mut output = input.clone();
        for layer in &self.layers {
            output = layer.calculate_output(&output);
        }
        output
    }

    /// Trains the given neural network using the provided dataset, batch size,
    /// number of epochs, and learning rate.
    pub fn train(
        &mut self,
        data: &Dataset,
        batch_size: usize,
        epochs: usize,
        learn_rate: f64,
    ) {
        for epoch in 1..=epochs {
            println!("Training epoch: {}", epoch);
            let (mut inputs, mut targets) = (
                Vec::<&Array1<f64>>::new(),
                Vec::<&Array1<f64>>::new()
            );
            let total = data.inputs.len();
            let mut processed = 0;
            let mut last_progress = 0;
            for (input, target) in data.random_iterator() {
                inputs.push(input);
                targets.push(target);
                processed += 1;
                if inputs.len() >= batch_size {
                    self.backwards_propagation(learn_rate, &inputs, &targets);
                    inputs.clear();
                    targets.clear();
                }
                // Print progress bar every 1%
                let progress = (processed * 100) / total;
                if progress >= last_progress + 1 || processed == total {
                    let bar_len = 30;
                    let filled = (progress * bar_len) / 100;
                    let bar: String = "#".repeat(filled) + &"-".repeat(bar_len - filled);
                    print!("\r[{}] {}%", bar, progress);
                    use std::io::Write;
                    std::io::stdout().flush().unwrap();
                    last_progress = progress;
                }
            }
            self.backwards_propagation(learn_rate, &inputs, &targets);
            println!("\r[{}] 100%", "#".repeat(30));
        }
    }

    /// Performs backward propagation using gradients calculated with the chain rule.
    /// It updates the weights and biases of the network based on the given inputs and targets batch.
    pub fn backwards_propagation(&mut self, learning_rate: f64, inputs: &Vec<&Array1<f64>>, targets: &Vec<&Array1<f64>>) {
        assert!(targets.len() == inputs.len(), "Targets len {} != Inputs len {}", targets.len(), inputs.len());
        let mut new_layers = self.layers.clone();
        let n = inputs.len();

        for i in 0..inputs.len() {
            let (input, target) = (&inputs[i], &targets[i]);
            assert!(target.len() == self.layers.last().unwrap().dim, "Target len {} != {}", target.len(), self.layers.last().unwrap().dim);
            assert!(input.len() == self.input_dim, "Input len {} != {}", input.len(), self.input_dim);

            // Calculate the gradient for the given batch with $\frac{1}{N} \nu \ * \sum_{i=1}^{N} \grad C_i$
            let gradients = self.calculate_gradient(input, target);
            for l in 0..self.layers.len() {
                let (grad_b, grad_w) = &gradients[l];
                new_layers[l].bias = &new_layers[l].bias - learning_rate * grad_b * (1.0 / n as f64);
                new_layers[l].weights = &new_layers[l].weights - learning_rate * grad_w * (1.0 / n as f64);
            }
        }

        self.layers = new_layers;
    }

    /// Calculates the gradient for each of the layers of the network in the format
    /// Vec<(Array1<f64>, Array2<f64>)> where every first entry of the tuple are the partial derivatives
    /// of the biases and the second entry of the tuple are the partial derivatives of the weights.
    fn calculate_gradient(&self, input: &Array1<f64>, target: &Array1<f64>) -> Vec<(Array1<f64>, Array2<f64>)> {
        // We calculate the output and store all the intermediate results before
        // applying the activation functions $z^{(l)}_i$ in a list z and the results
        // after applying the activation func $a^{(l)}_i$ in a list a
        let (a, z) = self.calculate_a_and_z(input);

        // The variable we will be returning
        let mut grad_output = Vec::<(Array1<f64>, Array2<f64>)>::new();

        // Now we compute $\partial C / \partial a^{(L)}$
        // with C the cost/loss function and L the final layer
        let output = a.last().unwrap();
        let mut grad_a: Array1<f64> = self.loss.gradient(&output, &target);

        // Now we loop through each of the layers backwards, calculating three gradient vectors each time
        for l in (0..self.layers.len()).rev() {
            // Calculation of the gradient for weights: $\partial C / \partial w_{jk}^{(l)}$
            // and of the gradient for the biases: $\partial C / \partial b^{(l)}$
            let grad_w = self.calculate_weight_gradient(l, &a, &z, &grad_a, &input);
            let grad_b = self.calculate_bias_gradient(l, &z, &grad_a);
            grad_output.push((grad_b, grad_w));

            // Calculation of then new gradient on the current layer: $\partial C / \partial a^{(l)}$
            if l > 0 {
                grad_a = self.calculate_a_gradient(l - 1, &z, &grad_a);
            }
        }

        grad_output.reverse();
        grad_output
    }

    /// Private function to help calculate the a and z vectors for every layer.
    /// So this function performs the forward step of the backpropagation algorithm.
    fn calculate_a_and_z (&self, input: &Array1<f64>) -> (Vec<Array1<f64>>, Vec<Array1<f64>>) {
        let mut z = Vec::<Array1<f64>>::new();
        let mut a = Vec::<Array1<f64>>::new();
        let mut output = input.clone();
        for layer in self.layers.iter() {
            let z_l = layer.calculate_output_no_activation(&output);
            output = z_l.mapv(|x| layer.activation.func(x));
            a.push(output.clone());
            z.push(z_l);

        }
        (a, z)
    } 

    /// Private function to help calculate the weight gradient for a given layer.
    /// This function computes $\partial C / \partial w^{(l)}$.
    fn calculate_weight_gradient(
        &self,
        l: usize,
        a: &Vec<Array1<f64>>,
        z: &Vec<Array1<f64>>,
        grad_a: &Array1<f64>,
        input: &Array1<f64>,
    ) -> Array2<f64> {
        let layer = &self.layers[l];
        let mut grad_w = Array2::<f64>::zeros(layer.weights.dim());
        for j in 0..layer.weights.nrows() {
            for k in 0..layer.weights.ncols() {
                grad_w[[j, k]] = if l > 0 {
                    a[l-1][k] * layer.activation.derivative(z[l][j]) * grad_a[j]
                } else {
                    input[k] * layer.activation.derivative(z[l][j]) * grad_a[j]
                };
            }
        }
        grad_w
    }

    /// Private function to help calculate the bias gradient for a given layer.
    /// This function computes $\partial C / \partial b^{(l)}$.
    fn calculate_bias_gradient(&self, l: usize, z: &Vec<Array1<f64>>, grad_a: &Array1<f64>) -> Array1<f64> {
        let layer = &self.layers[l];
        let mut grad_b = Array1::<f64>::zeros(layer.dim);
            for k in 0..layer.dim {
                grad_b[k] = grad_a[k] * layer.activation.derivative(z[l][k]);
            }
        grad_b
    }

    /// Private function to help calculate the new a gradient for a given layer.
    /// This function computes $\partial C / \partial a^{(l)}$ using the chain rule.
    fn calculate_a_gradient(&self, l: usize, z: &Vec<Array1<f64>>, grad_a: &Array1<f64>) -> Array1<f64> {
        let layer = &self.layers[l];
        let mut new_grad_a = Array1::<f64>::zeros(layer.dim);
            for k in 0..layer.dim {
                // Calculate $\partial C / \partial a^{(l)}_k$
                // $\partial C / \partial a^{(l)}_k = \sum_j w_{jk}^{(l+1)} \sigma'(z_j^{(l+1)}) \partial C / \partial a^{(l+1)}_j$
                let mut sum: f64 = 0.0;
                for j in 0..grad_a.dim() {
                    sum += layer.weights[[j, k]] * layer.activation.derivative(z[l][j]) * grad_a[j];
                }
                new_grad_a[k] = sum;
            }
        new_grad_a
    }
}

#[cfg(test)]
mod tests {
    use std::vec;

    use super::*;
    use ndarray::{arr1, arr2};
    use crate::rnet::activation::Activation;

    #[test]
    fn test_network_forward() {
        // Create a network with three layers of dimension 2
        let layer1 = Layer::new(
            2,
            Activation::Tanh,
            arr1(&[0.0, 0.0]),
            arr2(&[[1.0, 0.0], [0.0, 1.0]]),
        );
        let layer2 = layer1.clone();
        let network = Network::new(2, vec![layer1, layer2], Loss::MSE);

        // Test the forward propagation
        let input = arr1(&[1.0, 1.0]);
        let output = network.forward_prop(&input);
        assert_eq!(output, arr1(&[(1.0 as f64).tanh().tanh(), (1.0 as f64).tanh().tanh()]));
    }

    #[test]
    fn test_network_backprop_and_gate() {
        // Create a network with two layers, one with dimension 2 and one with dimension 1
        let layer1 = Layer::new(
            1,
            Activation::ReLu,
            arr1(&[0.0]),
            arr2(&[[1.0, 1.0]]),
        );
        let mut network = Network::new(2, vec![layer1], Loss::MSE);

        // Create all possible input combinations for the AND gate
        // replicating the one with output one three times to
        // have a balanced dataset
        let dataset = vec![
            (arr1(&[0.0, 0.0]), arr1(&[0.0])),
            (arr1(&[0.0, 1.0]), arr1(&[0.0])),
            (arr1(&[1.0, 0.0]), arr1(&[0.0])),
            (arr1(&[1.0, 1.0]), arr1(&[1.0])),
            (arr1(&[1.0, 1.0]), arr1(&[1.0])),
            (arr1(&[1.0, 1.0]), arr1(&[1.0])),
        ];

        // Learn the AND gate
        let runs = 10;
        for _ in 0..runs {
            for (input, target) in &dataset {
                network.backwards_propagation(0.1, &vec![&input], &vec![&target]);
            }
        }

        // Test the forward propagation after training
        for (input, target) in &dataset {
            let test_output = network.forward_prop(&input);
            assert!(test_output[0].round() == target[0]);
        }
    }
}