use std::vec;

use crate::rnet::data::Dataset;
use crate::rnet::network::Network;

/// Computes the mean squared errors between the network's predictions and the target outputs.
pub fn mean_squared_errors(network: &Network, dataset: &Dataset) -> Vec<f64> {
    let mut squared_errors = vec![0.0; dataset.len()];
    for (input, target) in dataset.iterator() {
        let test_output = network.forward_prop(&input);
        squared_errors.push(
            (test_output - target).mapv(|x| x.powi(2) / target.len() as f64).sum()
        );
    }
    squared_errors
}
