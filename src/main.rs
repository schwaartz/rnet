#![allow(dead_code)]

mod rnet;

use rnet::layer::*;
use rnet::network::*;
use rnet::activation::*;
use rnet::loss::*;
use rnet::data::*;
use rnet::rnet::*;

use ndarray::{arr1, arr2};

fn main() {
    let _layer = Layer::new(
        3,
        Activation::Sigmoid,
        arr1(&[0.0, 0.0, 0.0]),
        arr2(&[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
    );
    let _output = _layer.calculate_output(&arr1(&[0.0, 0.0, 0.0]));
    let _nn = Network::new(3, vec![_layer], Loss::MSE);
    let _nn_output = _nn.forward_prop(&arr1(&[0.0, 0.0, 0.0]));
    println!("Created a neural network and a layer + did some calculations");
}
