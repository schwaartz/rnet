use ndarray::{Array1, Array2};
use crate::layer::Layer;
use crate::loss::Loss;

/// The Network struct implements a simple neural network
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

    /// Performs a backward propagation step using gradients calculated with the chain rule
    pub fn backwards_propagation(&mut self, learning_rate: f64, input: &Array1<f64>, target: &Array1<f64>) {
        let gradients = self.calculate_gradient(input, target);
        for l in 0..self.layers.len() {
            let (grad_b, grad_w) = &gradients[l];
            self.layers[l].bias = &self.layers[l].bias - learning_rate * grad_b;
            self.layers[l].weights = &self.layers[l].weights - learning_rate * grad_w;
        }
    }

    /// Calculates the gradient for each of the layers of the network in the format
    /// Vec<(Array1<f64>, Array2<f64>)> where every first entry of the tuple are the partial derivatives
    /// of the biases and the second entry of the tuple are the partial derivatives of the weights.
    fn calculate_gradient(&self, input: &Array1<f64>, target: &Array1<f64>) -> Vec<(Array1<f64>, Array2<f64>)> {
        assert!(target.len() == self.layers.last().unwrap().dim);
        assert!(input.len() == self.input_dim);

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
            grad_a = self.calculate_new_a_gradient(l, &z, &grad_a);
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
        for k in 0..layer.weights.nrows() {
            for j in 0..layer.weights.ncols() {
                grad_w[[k, j]] = if l > 0 {
                    a[l-1][j] * layer.activation.derivative(z[l][k]) * grad_a[k]
                } else {
                    input[j] * layer.activation.derivative(z[l][k]) * grad_a[k]
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
    fn calculate_new_a_gradient(&self, l: usize, z: &Vec<Array1<f64>>, grad_a: &Array1<f64>) -> Array1<f64> {
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
    use super::*;
    use ndarray::{arr1, arr2};
    use crate::activation::Activation;

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
    fn test_network_backprop() {
        // Create a network with three layers of dimension 2
        let layer1 = Layer::new(
            2,
            Activation::ReLu,
            arr1(&[0.0, 0.0]),
            arr2(&[[1.0, 0.0], [0.0, 1.0]]),
        );
        let layer2 = layer1.clone();
        let mut network = Network::new(2, vec![layer1, layer2], Loss::MSE);

        // Test whether the backpropagation changed the values accordingly
        let input = arr1(&[1.0, 1.0]);
        let target = arr1(&[2.0, 0.0]);
        network.backwards_propagation(1.0, &input, &target);

        // We know that this will output [1.0, 1.0], so we know that the bias of the
        // first neuron in the output layer should have increased. The other weights
        // and biases are a lot more complicated.
        assert!(network.layers[1].bias[0] > 0.0);
    }
}