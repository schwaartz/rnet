use ndarray::Array1;
use crate::activation::Activation;

/// The sigmoid activation function
#[derive(Debug)]
pub struct Sigmoid;

impl Activation for Sigmoid {
    fn compute_value(&self, x: &Array1<f32>) -> Array1<f32> {
        x.mapv(|x| 1.0 / (1.0 + (-x).exp()))
    }

    fn compute_derivative(&self, x: &Array1<f32>) -> Array1<f32> {
        let s = self.compute_value(x);
        &s * (1.0 - &s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_sigmoid_compute_value_vector() {
        let sigmoid = Sigmoid;
        let input = array![-1.0, 0.0, 1.0];
        let output = sigmoid.compute_value(&input);
        let expected = array![0.26894143, 0.5, 0.7310586];
        for (o, e) in output.iter().zip(expected.iter()) {
            assert!((o - e).abs() < 1e-6);
        }
    }

    #[test]
    fn test_sigmoid_compute_derivative_vector() {
        let sigmoid = Sigmoid;
        let input = array![-1.0, 0.0, 1.0];
        let output = sigmoid.compute_derivative(&input);
        let expected = array![0.19661193, 0.25, 0.19661193];
        for (o, e) in output.iter().zip(expected.iter()) {
            assert!((o - e).abs() < 1e-6);
        }
    }
}