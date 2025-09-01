use ndarray::Array1;
use crate::activation::Activation;

/// The linear activation function
#[derive(Debug)]
pub struct Linear;

impl Activation for Linear {
    fn compute_value(&self, x: &Array1<f32>) -> Array1<f32> {
        x.clone()
    }

    fn compute_derivative(&self, x: &Array1<f32>) -> Array1<f32> {
        x.mapv(|_| 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_linear_compute_value() {
        let linear = Linear;
        let input = array![1.0, 2.0, 3.0];
        let expected = array![1.0, 2.0, 3.0];
        assert_eq!(linear.compute_value(&input), expected);
    }

    #[test]
    fn test_linear_compute_derivative() {
        let linear = Linear;
        let input = array![1.0, 2.0, 3.0];
        let expected = array![1.0, 1.0, 1.0];
        assert_eq!(linear.compute_derivative(&input), expected);
    }
}