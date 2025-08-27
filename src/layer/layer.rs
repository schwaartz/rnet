use ndarray::{Array1, Array2};

/// The Layer struct structs represents any (non-input) layer in a NN
/// Input layers do not require weights or a `calculate` function or a bias
/// as they only receive input data.
pub struct Layer {
    pub num_nodes: usize,
    pub activation: fn(f64) -> f64,
    pub bias: Array1<f64>,
    pub weights: Array2<f64>, // Weights connecting to the previous layer
}

impl Layer {
    /// Creates a new Layer
    pub fn new(
        num_nodes: usize,
        activation: fn(f64) -> f64,
        bias: Array1<f64>,
        weights: Array2<f64>,
    ) -> Self {
        Layer {
            num_nodes,
            activation,
            bias,
            weights,
        }
    }

    /// Calculates the output of the layer given the outputs of the previous layer (input vector).
    /// It also checks whether or not the input size matches the weight matrix and the bias in advance.
    pub fn calculate(&self, input: &Array1<f64>) -> Array1<f64> {
        assert!(input.len() == self.weights.ncols());
        assert!(self.weights.nrows() == self.num_nodes);
        assert!(self.bias.len() == self.num_nodes);

        let z = self.weights.dot(input) + &self.bias;
        z.mapv(self.activation)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, arr2};

    #[test]
    fn test_calculate() {
        let bias = arr1(&[1.0, 0.0, 0.0]);
        let mtx = arr2(&[[1.0, 2.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]); // Weight matrix
        let layer = Layer::new(3, |x| x.tanh(), bias, mtx);
        let input = arr1(&[1.0, 1.0, 1.0]);
        let output = layer.calculate(&input);
        assert_eq!(output, arr1(&[(4.0 as f64).tanh(), (2.0 as f64).tanh(), (3.0 as f64).tanh()]));
    }
}