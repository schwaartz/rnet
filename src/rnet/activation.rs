
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
}