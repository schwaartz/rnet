use crate::loss::Loss;
use ndarray::Array1;

/// A struct for the mean squared error loss
pub struct MSE;

impl Loss for MSE {
    /// Computes the value of the mean squared error loss
    fn compute_value(&self, output: &Array1<f32>, target: &Array1<f32>) -> f32 {
        let diff = output - target;
        diff.dot(&diff) / 2.0
    }

    /// Computes the gradient of the mean squared error loss
    fn compute_gradient(&self, output: &Array1<f32>, target: &Array1<f32>) -> Array1<f32> {
        output - target
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mse_loss() {
        let output = Array1::from_vec(vec![0.0, 1.0, 2.0]);
        let target = Array1::from_vec(vec![0.0, 1.0, 1.0]);
        let mse = MSE;

        let loss = mse.compute_value(&output, &target);
        let gradient = mse.compute_gradient(&output, &target);

        assert_eq!(loss, 0.5);
        assert_eq!(gradient, Array1::from_vec(vec![0.0, 0.0, 1.0]));
    }
}