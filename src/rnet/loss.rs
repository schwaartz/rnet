use ndarray::Array1;

/// Represents the different loss functions available
pub enum Loss {
    MSE,
    CrossEntropy,
}

impl Loss {
    /// Computes the loss between the output and target arrays.
    pub fn loss(&self, output: &Array1<f64>, target: &Array1<f64>) -> f64 {
        match self {
            Loss::MSE => mse_loss(output, target),
            Loss::CrossEntropy => cross_entropy_loss(output, target),
        }
    }

    /// Computes the gradient between the output and target arrays.
    /// BE CAREFUL TO PASS THE CORRECT OUTPUT (logits and probabilities should not be confused).
    /// For more information, look at the documentation added to the implementation of the loss function.
    pub fn gradient(&self, output: &Array1<f64>, target: &Array1<f64>) -> Array1<f64> {
        match self {
            Loss::MSE => mse_gradient(output, target),
            Loss::CrossEntropy => cross_entropy_gradient(output, target),
        }
    }
}

/// Compute \sum_i \frac{1}{2} (\hat{y}_i - y_i)^2.
/// Assumed the raw output logits as the output.
fn mse_loss(output: &Array1<f64>, target: &Array1<f64>) -> f64 {
    let diff = output - target;
    diff.dot(&diff) / 2.0
}

/// Compute \hat{y} - y.
/// Assumed the raw output logits as the output.
fn mse_gradient(output: &Array1<f64>, target: &Array1<f64>) -> Array1<f64> {
    output - target
}

const EPSILON: f64 = 1e-15; // Avoids log(0) = -inf from ruining the sum

/// Compute -\sum_i y_i \ln(\hat{y}_i).
/// Note that it is assumed that the outputs are assumed to be probabilities
/// (i.e. not the raw output of the neural network, but after softmax).
fn cross_entropy_loss(output: &Array1<f64>, target: &Array1<f64>) -> f64 {
    -target
        .iter()
        .zip(output.iter())
        .map(|(t, o)| t * (o + EPSILON).ln())
        .sum::<f64>()
}

/// Compute \hat{y} - y.
/// Note that it is assumed that the outputs are assumed to be logits
/// (i.e. the raw output of the neural network, before softmax).
/// This is done to ensure that the gradient is computed in a numerically stable way
fn cross_entropy_gradient(output: &Array1<f64>, target: &Array1<f64>) -> Array1<f64> {
    output - target
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mse_loss() {
        let output = Array1::from_vec(vec![0.0, 1.0, 2.0]);
        let target = Array1::from_vec(vec![0.0, 1.0, 1.0]);
        let loss = Loss::MSE;
        assert_eq!(loss.loss(&output, &target), 0.5);
    }

    #[test]
    fn test_mse_gradient() {
        let output = Array1::from_vec(vec![0.0, 1.0, 2.0]);
        let target = Array1::from_vec(vec![0.0, 1.0, 1.0]);
        let loss = Loss::MSE;
        assert_eq!(loss.gradient(&output, &target), Array1::from_vec(vec![0.0, 0.0, 1.0]));
    }

    #[test]
    fn test_cross_entropy_loss() {
        let output = Array1::from_vec(vec![0.1, 0.9, 0.0]); // Probability inputs
        let target = Array1::from_vec(vec![0.0, 1.0, 0.0]);
        let loss = Loss::CrossEntropy;
        assert_eq!(loss.loss(&output, &target), -(0.9 + EPSILON as f64).ln());
    }

    #[test]
    fn test_cross_entropy_gradient() {
        let output = Array1::from_vec(vec![1.0, 5.0, 0.0]); // Logit inputs
        let target = Array1::from_vec(vec![0.0, 1.0, 0.0]);
        let loss = Loss::CrossEntropy;
        assert_eq!(loss.gradient(&output, &target), Array1::from_vec(vec![1.0, 4.0, 0.0]));
    }
}