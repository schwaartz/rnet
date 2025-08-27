pub trait Activation {
    fn func(x: f64) -> f64;
    fn derivative(x: f64) -> f64;
}

/// A struct representing the Sigmoid activation function
pub struct Sigmoid {}

impl Activation for Sigmoid {
    /// Sigmoid activation function
    fn func(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    /// Sigmoid derivative function
    fn derivative(x: f64) -> f64 {
        let s = Self::func(x);
        s * (1.0 - s)
    }
}

/// A struct representing the Hyperbolic Tangent activation function
pub struct Tanh {}

impl Activation for Tanh {
    /// Hyperbolic tangent activation function
    fn func(x: f64) -> f64 {
        x.tanh()
    }

    /// Hyperbolic tangent derivative function
    fn derivative(x: f64) -> f64 {
        let t = Self::func(x);
        1.0 - t * t
    }
}

#[cfg(test)]
mod tests{
    use super::*;

    #[test]
    fn test_sigmoid() {
        assert_eq!(Sigmoid::func(0.0), 0.5);
        assert_eq!(Sigmoid::func(1.0), 0.7310585786300049);
    }

    #[test]
    fn test_sigmoid_derivative() {
        assert_eq!(Sigmoid::derivative(0.0), 0.25);
        assert_eq!(Sigmoid::derivative(1.0), 0.19661193324148185);
    }

    #[test]
    fn test_tanh() {
        assert_eq!(Tanh::func(0.0), 0.0);
        assert_eq!(Tanh::func(1.0), 0.7615941559557649);
    }

    #[test]
    fn test_tanh_derivative() {
        assert_eq!(Tanh::derivative(0.0), 1.0);
        assert_eq!(Tanh::derivative(1.0), 0.41997434161402614);
    }
}