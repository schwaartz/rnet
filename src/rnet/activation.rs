use ndarray::Array1;

/// An enum representing different types of activation functions
#[derive(Clone, Copy)]
pub enum Activation {
    Sigmoid,
    Tanh,
    ReLu,
}

impl Activation {
    pub fn func(&self, x: f64) -> f64 {
        match self {
            Activation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            Activation::Tanh => x.tanh(),
            Activation::ReLu => x.max(0.0),
        }
    }

    pub fn derivative(&self, x: f64) -> f64 {
        match self {
            Activation::Sigmoid => {
                let s = 1.0 / (1.0 + (-x).exp());
                s * (1.0 - s)
            }
            Activation::Tanh => {
                let t = x.tanh();
                1.0 - t * t
            }
            Activation::ReLu => {
                if x > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
        }
    }
}

/// An enum representing different types of output activation functions
pub enum OutputActivation {
    Softmax,
}

impl OutputActivation {
    /// Applies the activation function to the input array.
    pub fn func(&self, x: Array1<f64>) -> Array1<f64> {
        match self {
            OutputActivation::Softmax => {
                let exp_x = x.mapv(|v| v.exp());
                let sum = exp_x.sum();
                exp_x / sum
            }
        }
    }
}




#[cfg(test)]
mod tests{
    use super::*;

    #[test]
    fn test_sigmoid() {
        let activation = Activation::Sigmoid;
        assert_eq!(activation.func(0.0), 0.5);
        assert_eq!(activation.func(1.0), 0.7310585786300049);
    }

    #[test]
    fn test_sigmoid_derivative() {
        let activation = Activation::Sigmoid;
        assert_eq!(activation.derivative(0.0), 0.25);
        assert_eq!(activation.derivative(1.0), 0.19661193324148185);
    }

    #[test]
    fn test_tanh() {
        let activation = Activation::Tanh;
        assert_eq!(activation.func(0.0), 0.0);
        assert_eq!(activation.func(1.0), 0.7615941559557649);
    }

    #[test]
    fn test_tanh_derivative() {
        let activation = Activation::Tanh;
        assert_eq!(activation.derivative(0.0), 1.0);
        assert_eq!(activation.derivative(1.0), 0.41997434161402614);
    }

    #[test]
    fn test_relu() {
        let activation = Activation::ReLu;
        assert_eq!(activation.func(0.0), 0.0);
        assert_eq!(activation.func(1.0), 1.0);
    }

    #[test]
    fn test_relu_derivative() {
        let activation = Activation::ReLu;
        assert_eq!(activation.derivative(0.0), 0.0);
        assert_eq!(activation.derivative(1.0), 1.0);
    }

    #[test]
    fn test_softmax() {
        let output_activation = OutputActivation::Softmax;
        let input = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let expected = Array1::from_vec(vec![0.09003057317038046, 0.24472847105479767, 0.6652409557748219]);
        assert_eq!(output_activation.func(input), expected);
    }
}