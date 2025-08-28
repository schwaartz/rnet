use ndarray::Array1;

/// Represents the different loss functions available
pub enum Loss {
    MSE,
    CrossEntropy,
}

impl Loss {
    /// Computes the loss between the output and target arrays
    pub fn loss(&self, output: &Array1<f64>, target: &Array1<f64>) -> f64 {
        match self {
            Loss::MSE => mse_loss(output, target),
            Loss::CrossEntropy => unimplemented!(),
        }
    }

    /// Computes the gradient between the output and target arrays
    pub fn gradient(&self, output: &Array1<f64>, target: &Array1<f64>) -> Array1<f64> {
        match self {
            Loss::MSE => mse_gradient(output, target),
            Loss::CrossEntropy => unimplemented!(),
        }
    }
}

/// Compute \sum_i \frac{1}{2} (\hat{y}_i - y_i)^2
fn mse_loss(output: &Array1<f64>, target: &Array1<f64>) -> f64 {
    let diff = output - target;
    diff.dot(&diff) / 2.0
}

/// Compute \hat{y}_i - y_i
fn mse_gradient(output: &Array1<f64>, target: &Array1<f64>) -> Array1<f64> {
    output - target
}