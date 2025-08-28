use std::vec;

use crate::rnet::data::Dataset;
use crate::rnet::network::Network;
use ndarray::Array1;

/// Trains the given neural network using the provided dataset, batch size,
/// number of epochs, and learning rate.
pub fn train_network(
    net: &mut Network,
    data: Dataset,
    batch_size: usize,
    epochs: usize,
    learn_rate: f64,
) {
    for _ in 0..epochs {
        let (mut inputs, mut targets) = (
            Vec::<&Array1<f64>>::new(),
            Vec::<&Array1<f64>>::new()
        );
        for (input, target) in data.random_iterator() {
            inputs.push(input);
            targets.push(target);
            if inputs.len() >= batch_size {
                net.backwards_propagation(learn_rate, &inputs, &targets);
                inputs.clear();
                targets.clear();
            }
        }
        net.backwards_propagation(learn_rate, &inputs, &targets);
    }
}

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
