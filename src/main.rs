#![allow(dead_code)]

fn main() {
    let weights = vec![vec![0.5, -0.2], vec![0.1, 0.4]];
    let bias = vec![0.0, 0.0];

    let relu_layer = Layer::new(weights.clone(), bias.clone(), Box::new(ReLU));
    let sigmoid_layer = Layer::new(weights.clone(), bias.clone(), Box::new(Sigmoid));

    let input = vec![1.0, 2.0];

    println!("ReLU output: {:?}", relu_layer.forward(&input));
    println!("Sigmoid output: {:?}", sigmoid_layer.forward(&input));
}