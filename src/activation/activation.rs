use std::fmt::Debug;
use ndarray::Array1;

/// A trait that represents an activation function with 1D arrays as its input
pub trait Activation: Debug {
    fn compute_value(&self, x: &Array1<f32>) -> Array1<f32>;
    fn compute_derivative(&self, x: &Array1<f32>) -> Array1<f32>;
}