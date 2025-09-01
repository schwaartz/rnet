use ndarray::Array1;
use crate::activation::Activation;

/// The rectified linear unit (ReLU) activation function
#[derive(Debug)]
pub struct ReLU;

impl Activation for ReLU {
    fn compute_value(&self, x: &Array1<f32>) -> Array1<f32> {
        x.mapv(|x| x.max(0.0))
    }

    fn compute_derivative(&self, x: &Array1<f32>) -> Array1<f32> {
        x.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    
    #[test]
    fn test_relu_compute_value_positive() {
        let relu = ReLU;
        let input = array![1.0, 2.5, 0.0, 3.3];
        let expected = array![1.0, 2.5, 0.0, 3.3];
        assert_eq!(relu.compute_value(&input), expected);
    }

    #[test]
    fn test_relu_compute_value_negative() {
        let relu = ReLU;
        let input = array![-1.0, -2.5, 0.0, 3.3];
        let expected = array![0.0, 0.0, 0.0, 3.3];
        assert_eq!(relu.compute_value(&input), expected);
    }

    #[test]
    fn test_relu_compute_derivative_positive() {
        let relu = ReLU;
        let input = array![1.0, 2.5, 0.0, 3.3];
        let expected = array![1.0, 1.0, 0.0, 1.0];
        assert_eq!(relu.compute_derivative(&input), expected);
    }

    #[test]
    fn test_relu_compute_derivative_negative() {
        let relu = ReLU;
        let input = array![-1.0, -2.5, 0.0, 3.3];
        let expected = array![0.0, 0.0, 0.0, 1.0];
        assert_eq!(relu.compute_derivative(&input), expected);
    }
}