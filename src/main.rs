#![allow(dead_code)]

mod rnet;

use rnet::activation::*;
use rnet::data::*;
use rnet::*;

use ndarray::arr1;

fn main() {
    let mut rnet = RNet::new_default_nn(vec![4, 4, 1], vec![Activation::Sigmoid, Activation::Sigmoid]);
    let train_inputs = vec![
        arr1(&[0.0, 0.0, 0.0, 0.0]),
        arr1(&[1.0, 0.0, 0.0, 0.0]),
        arr1(&[0.0, 1.0, 0.0, 0.0]),
        arr1(&[0.0, 0.0, 1.0, 0.0]),
        arr1(&[0.0, 0.0, 0.0, 1.0]),
    ];
    let train_targets = vec![
        arr1(&[0.0]),
        arr1(&[0.0]),
        arr1(&[1.0]),
        arr1(&[1.0]),
        arr1(&[0.0]),
    ];
    let train_data = Dataset::new(train_inputs, train_targets);
    rnet.set_epochs(10000);
    rnet.set_batch_size(2);
    rnet.set_learning_rate(1.0);
    rnet.train(&train_data);

    let test_inputs = vec![
        arr1(&[0.0, 0.0, 0.0, 0.0]),
        arr1(&[1.0, 0.0, 0.0, 0.0]),
        arr1(&[0.0, 1.0, 0.0, 0.0]),
        arr1(&[0.0, 0.0, 1.0, 0.0]),
        arr1(&[0.0, 0.0, 0.0, 1.0]),
    ];
    let test_targets = vec![
        arr1(&[0.0]),
        arr1(&[0.0]),
        arr1(&[1.0]),
        arr1(&[1.0]),
        arr1(&[0.0]),
    ];
    let test_data = Dataset::new(test_inputs, test_targets);
    let mse = rnet.mean_squared_error(&test_data);
    println!("Mean Squared Error on Test Data: {}", mse);
    println!("Final layer weights: {:?}", rnet.network.layers.last().unwrap().weights);
    println!("Final layer biases: {:?}", rnet.network.layers.last().unwrap().bias);
    println!("Hidden layer weights: {:?}", rnet.network.layers[0].weights);
    println!("Hidden layer biases: {:?}", rnet.network.layers[0].bias);
}
