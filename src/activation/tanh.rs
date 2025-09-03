use ndarray::Array1;
use crate::activation::Activation;

/// The hyperbolic tangent (tanh) activation function
#[derive(Debug)]
pub struct Tanh;

impl Activation for Tanh {
    fn compute_value(&self, x: &Array1<f32>) -> Array1<f32> {
        x.mapv(|x| x.tanh())
    }

    fn compute_derivative(&self, x: &Array1<f32>) -> Array1<f32> {
        x.mapv(|x| 1.0 - x.tanh().powi(2))
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tanh_compute_value() {
        let tanh = Tanh;
        let input = Array1::from_vec(vec![-1.0, 0.0, 1.0]);
        let output = tanh.compute_value(&input);
        let expected = Array1::from_vec(vec![
            (-1.0f32).tanh(),
            0.0f32.tanh(),
            1.0f32.tanh(),
        ]);
        assert_eq!(output, expected);
    }

    #[test]
    fn test_tanh_compute_derivative() {
        let tanh = Tanh;
        let input = Array1::from_vec(vec![-1.0, 0.0, 1.0]);
        let output = tanh.compute_derivative(&input);
        let expected = Array1::from_vec(vec![
            1.0 - (-1.0f32).tanh().powi(2),
            1.0 - 0.0f32.tanh().powi(2),
            1.0 - 1.0f32.tanh().powi(2),
        ]);
        assert_eq!(output, expected);
    }
}
