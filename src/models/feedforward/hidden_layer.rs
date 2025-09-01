use crate::activation::Activation;
use ndarray::{Array1, Array2};

pub struct HiddenLayer {
    pub dim: usize,
    pub weights: Option<Array2<f32>>,
    pub biases: Option<Array1<f32>>,
    pub activation: Box<dyn Activation>,
}

impl HiddenLayer {
    /// Generates a new layer struct with uninitialized weights and biases
    pub fn new(dim: usize, activation: Box<dyn Activation>) -> Self {
        Self { dim, weights: None, biases: None, activation }
    }

    /// Takes ownership of the weights and gives it to the layer
    pub fn set_weights(&mut self, weights: Array2<f32>) {
        if self.biases.is_some() {
            assert_eq!(
                weights.nrows(),
                self.biases.as_ref().unwrap().len(),
                "Weights rows ({}) must match biases length ({})",
                weights.nrows(),
                self.biases.as_ref().unwrap().len()
            );
        }
        self.weights = Some(weights);
    }

    /// Takes ownership of the biases and gives it to the layer
    pub fn set_biases(&mut self, biases: Array1<f32>) {
        if self.weights.is_some() {
            assert_eq!(
                biases.len(),
                self.weights.as_ref().unwrap().nrows(),
                "Biases length ({}) must match weights rows ({})",
                biases.len(),
                self.weights.as_ref().unwrap().nrows()
            );
        }
        self.biases = Some(biases);
    }

    /// Computes the output of the layer for a given input
    pub fn compute_output(&self, input: &Array1<f32>) -> Array1<f32> {
        let logits = self.compute_logits(input);
        self.activation.compute_value(&logits)
    }

    /// Computes the logits of the layer for a given input
    pub fn compute_logits(&self, input: &Array1<f32>) -> Array1<f32> {
        assert!(self.weights.is_some(), "Weights must be Some");
        assert!(self.biases.is_some(), "Biases must be Some");

        let weights = self.weights.as_ref().unwrap();
        let biases = self.biases.as_ref().unwrap();

        weights.dot(input) + biases
    }
}

#[cfg(test)]
mod tests{
    use crate::activation::Sigmoid;

    use super::*;
    use ndarray::array;

    #[test]
    fn test_new_layer_initialization() {
        let layer = HiddenLayer::new(4, Box::new(Sigmoid));
        assert_eq!(layer.dim, 4);
        assert!(layer.weights.is_none());
        assert!(layer.biases.is_none());
    }

    #[test]
    fn test_set_weights_and_biases() {
        let mut layer = HiddenLayer::new(2, Box::new(Sigmoid));
        let weights = array![[1.0, 2.0], [3.0, 4.0]];
        let biases = array![0.5, -0.5];
        layer.set_weights(weights.clone());
        layer.set_biases(biases.clone());
        assert_eq!(layer.weights.as_ref().unwrap(), &weights);
        assert_eq!(layer.biases.as_ref().unwrap(), &biases);
    }

    #[test]
    fn test_compute_logits() {
        let mut layer = HiddenLayer::new(2, Box::new(Sigmoid));
        let weights = array![[1.0, 2.0], [3.0, 4.0]];
        let biases = array![0.5, -0.5];
        layer.set_weights(weights);
        layer.set_biases(biases);
        let input = array![1.0, 2.0];
        let logits = layer.compute_logits(&input);
        assert_eq!(logits, array![5.5, 10.5]);
    }

    #[test]
    fn test_compute_output() {
        let mut layer = HiddenLayer::new(2, Box::new(Sigmoid));
        let weights = array![[1.0, 2.0], [3.0, 4.0]];
        let biases = array![0.5, -0.5];
        layer.set_weights(weights);
        layer.set_biases(biases);
        let input = array![1.0, 2.0];
        let output = layer.compute_output(&input);
        assert_eq!(output, Sigmoid.compute_value(&array![5.5, 10.5]));
    }

    #[test]
    #[should_panic(expected = "Weights must be Some")]
    fn test_compute_logits_without_weights() {
        let mut layer = HiddenLayer::new(2, Box::new(Sigmoid));
        let biases = array![0.5, -0.5];
        layer.set_biases(biases);
        let input = array![1.0, 2.0];
        let _ = layer.compute_logits(&input);
    }

    #[test]
    #[should_panic(expected = "Biases must be Some")]
    fn test_compute_logits_without_biases() {
        let mut layer = HiddenLayer::new(2, Box::new(Sigmoid));
        let weights = array![[1.0, 2.0], [3.0, 4.0]];
        layer.set_weights(weights);
        let input = array![1.0, 2.0];
        let _ = layer.compute_logits(&input);
    }

    #[test]
    #[should_panic(expected = "Biases length (3) must match weights rows (2)")]
    fn test_set_weights_and_biases_mismatch() {
        let mut layer = HiddenLayer::new(2, Box::new(Sigmoid));
        let weights = array![[1.0, 2.0], [3.0, 4.0]];
        let biases = array![0.5, -0.5, 1.0];
        layer.set_weights(weights);
        layer.set_biases(biases);
    }

    #[test]
    #[should_panic(expected = "Weights rows (2) must match biases length (3)")]
    fn test_set_biases_and_weights_mismatch() {
        let mut layer = HiddenLayer::new(2, Box::new(Sigmoid));
        let weights = array![[1.0, 2.0], [3.0, 4.0]];
        let biases = array![0.5, -0.5, 1.0];
        layer.set_biases(biases);
        layer.set_weights(weights);
    }
}