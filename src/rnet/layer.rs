use ndarray::{Array1, Array2};

use crate::rnet::activation::Activation;

/// The Layer struct structs represents any (non-input) layer in a NN
/// Input layers do not require weights or a `calculate` function or a bias
/// as they only receive input data.
#[derive(Debug, Clone)]
pub struct Layer {
    pub dim: usize,
    pub activation: Activation,
    pub bias: Array1<f64>,
    pub weights: Array2<f64>, // Weights connecting to the previous layer
}

impl Layer {
    /// Creates a new Layer
    pub fn new(
        dim: usize,
        activation: Activation,
        bias: Array1<f64>,
        weights: Array2<f64>,
    ) -> Self {
        Layer {
            dim,
            activation,
            bias,
            weights,
        }
    }

    /// Calculates the output a of the layer given the outputs of the previous layer (input vector).
    /// It also checks whether or not the input size matches the weight matrix and the bias in advance.
    pub fn calculate_output(&self, input: &Array1<f64>) -> Array1<f64> {
        assert!(input.len() == self.weights.ncols(), "Input len {} != {}", input.len(), self.weights.ncols());
        assert!(self.weights.nrows() == self.dim, "Weights rows {} != {}", self.weights.nrows(), self.dim);
        assert!(self.bias.len() == self.dim, "Bias len {} != {}", self.bias.len(), self.dim);

        let z = self.weights.dot(input) + &self.bias;
        z.mapv(|x| self.activation.func(x))
    }

    /// Calculates the output z of the layer without applying the activation function.
    /// This is useful for calculating the gradients during backpropagation
    pub fn calculate_output_no_activation(&self, input: &Array1<f64>) -> Array1<f64> {
        assert!(input.len() == self.weights.ncols(), "Input len {} != {}", input.len(), self.weights.ncols());
        assert!(self.weights.nrows() == self.dim, "Weights rows {} != {}", self.weights.nrows(), self.dim);
        assert!(self.bias.len() == self.dim, "Bias len {} != {}", self.bias.len(), self.dim);

        let z = self.weights.dot(input) + &self.bias;
        z
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, arr2};

    #[test]
    fn test_calculate_output() {
        let bias = arr1(&[1.0, 0.0, 0.0]);
        let mtx = arr2(&[[1.0, 2.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]); // Weight matrix
        let layer = Layer::new(3, Activation::Tanh, bias, mtx);
        let input = arr1(&[1.0, 1.0, 1.0]);
        let output = layer.calculate_output(&input);
        assert_eq!(output, arr1(&[(4.0 as f64).tanh(), (2.0 as f64).tanh(), (3.0 as f64).tanh()]));
    }
}