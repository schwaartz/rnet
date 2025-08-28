use ndarray::{Array1, Array2};
use crate::layer::Layer;
use crate::activation::Activation;
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

    /// Calculates the gradient for each of the layers of the network in the format
    /// Vec<(Array1<f64>, Array2<f64>)> where every first entry of the tuple are the partial derivatives
    /// of the biases and the second entry of the tuple are the partial derivatives of the weights.
    pub fn backward_prop(&self, output_grad: &Array1<f64>) -> Vec<(Array1<f64>, Array2<f64>)> {
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, arr2};

    #[test]
    fn test_network_forward() {
        let layer1 = Layer::new(
            2,
            Activation::Tanh,
            arr1(&[0.0, 0.0]),
            arr2(&[[1.0, 0.0], [0.0, 1.0]]),
        );
        let layer2 = layer1.clone();
        let network = Network::new(2, vec![layer1, layer2], Loss::MSE);
        let input = arr1(&[1.0, 1.0]);
        let output = network.forward_prop(&input);
        assert_eq!(output, arr1(&[(1.0 as f64).tanh().tanh(), (1.0 as f64).tanh().tanh()]));
    }
}