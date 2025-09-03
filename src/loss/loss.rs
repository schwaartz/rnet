use ndarray::Array1;

/// A trait for loss functions
pub trait Loss {
    fn compute_value(&self, output: &Array1<f32>, target: &Array1<f32>) -> f32;
    fn compute_gradient(&self, output: &Array1<f32>, target: &Array1<f32>) -> Array1<f32>;
}