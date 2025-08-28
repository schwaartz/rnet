use crate::rnet::data::Dataset;
use crate::rnet::network::Network;
use ndarray::Array1;

pub fn train_network(net: &mut Network, data: Dataset, batch_size: usize, epochs: usize, learn_rate: f64) {
    for _ in 0..epochs {
        let (mut inputs, mut targets) = (Vec::<&Array1<f64>>::new(), Vec::<&Array1<f64>>::new());
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