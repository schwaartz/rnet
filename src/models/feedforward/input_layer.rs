use ndarray::Array1;

/// A simple representation of an input layer. It has no activation function, weights, or biases.
pub struct InputLayer {
    pub dim: usize,
}

impl InputLayer {
    /// Creates a new InputLayer
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }

    /// Computes the output of the layer for a given input.
    /// It does nothing at the moment, yet it is still here for future use.
    pub fn compute_output(&self, input: &Array1<f32>) -> Array1<f32> {
        input.clone()
    }

    /// Computes the logits of the layer for a given input.
    /// It does nothing at the moment, yet it is still here for future use.
    pub fn compute_logits(&self, input: &Array1<f32>) -> Array1<f32> {
        input.clone()    
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_input_layer() {
        let layer = InputLayer::new(3);
        let input = Array1::from_vec(vec![1.0, 2.0, 3.0]);

        let output = layer.compute_output(&input);
        let logits = layer.compute_logits(&input);

        assert_eq!(output, input);
        assert_eq!(logits, input);
    }
}