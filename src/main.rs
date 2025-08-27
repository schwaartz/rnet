mod layer; mod network;
use crate::layer::Layer;
use crate::network::Network;
use ndarray::{arr1, arr2};

fn main() {
    let _layer = Layer::new(
        3,
        |x| x.tanh(),
        arr1(&[0.0, 0.0, 0.0]),
        arr2(&[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
    );
    let _output = _layer.calculate_output(&arr1(&[0.0, 0.0, 0.0]));
    let _nn = Network::new(3, vec![_layer]);
    let _nn_output = _nn.forward_prop(&arr1(&[0.0, 0.0, 0.0]));
    println!("Created a neural network and a layer + did some calculations");
}
