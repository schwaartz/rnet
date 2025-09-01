use crate::activation::Activation;
use ndarray::Array1;

/// The softmax activation function
#[derive(Debug)]
pub struct Softmax;

impl Activation for Softmax {
    fn compute_value(&self, x: &Array1<f32>) -> Array1<f32> {
        let exp_x = x.mapv(|x| x.exp());
        let sum = exp_x.sum();
        exp_x / sum
    }

    fn compute_derivative(&self, x: &Array1<f32>) -> Array1<f32> {
        let softmax = self.compute_value(x);
        &softmax * (1.0 - &softmax)
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_value() {
        let softmax = Softmax;
        let input = Array1::from_vec(vec![-1.0, 0.0, 1.0]);
        let output = softmax.compute_value(&input);

        let exp_vals: Array1<f32> = input.iter().map(|&x| x.exp()).collect();
        let sum: f32 = exp_vals.iter().sum();
        let expected: Array1<f32> = exp_vals.iter().map(|&x| x / sum).collect();

        assert_eq!(output, expected);
    }

    #[test]
    fn test_softmax_derivative() {
        let softmax = Softmax;
        let input = Array1::from_vec(vec![-1.0, 0.0, 1.0]);
        let output = softmax.compute_value(&input);
        let derivative = softmax.compute_derivative(&input);

        let expected = &output * (1.0 - &output);
        
        assert_eq!(derivative, expected);
    }
}
