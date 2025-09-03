use crate::loss::Loss;
use ndarray::Array1;

const EPSILON: f32 = 1e-12;

/// A struct for the cross-entropy loss
pub struct CrossEntropy;

impl Loss for CrossEntropy {
    /// Computes the value of the cross-entropy loss.
    /// A small EPSILON is added after applying the logarithm to avoid -inf bugs.
    fn compute_value(&self, output: &Array1<f32>, target: &Array1<f32>) -> f32 {
        -target.dot(&output.mapv(|x| x.ln() + EPSILON))
    }

    /// Computes the gradient of the cross-entropy loss
    fn compute_gradient(&self, output: &Array1<f32>, target: &Array1<f32>) -> Array1<f32> {
        output - target
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cross_entropy_loss() {
        let output = Array1::from_vec(vec![0.1, 0.9, 0.8]);
        let target = Array1::from_vec(vec![0.0, 1.0, 0.0]);
        let cross_entropy = CrossEntropy;

        let loss = cross_entropy.compute_value(&output, &target);
        let gradient = cross_entropy.compute_gradient(&output, &target);

        assert_eq!(loss, -target.dot(&output.mapv(|x| x.ln() + EPSILON)));
        assert_eq!(gradient, output - target);
    }
}