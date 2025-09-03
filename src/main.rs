#![allow(dead_code)]

use std::fs;
use image::ImageReader;
use ndarray::Array1;
use rnet::{activation::*, dataset::Dataset, loss::CrossEntropy, models::{FeedForward, HiddenLayer, InputLayer, OutputLayer}};

fn main() {
        let mut images = Vec::new();
    let mut labels = Vec::new();

    for digit in 0..10 {
        let train_folder = format!("{}/data/mnist/dataset/{}", std::env::current_dir().unwrap().display(), digit);
        println!("Loading data from {}", train_folder);
        for entry in fs::read_dir(&train_folder).unwrap() {
            let entry = entry.unwrap();
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("jpg") {
            let img = ImageReader::open(&path).unwrap().decode().unwrap().to_luma8();
            let flat: Array1<f32> = Array1::from(
                img.pixels()
                .map(|p| p[0] as f32 / 255.0)
                .collect::<Vec<_>>()
            );
            images.push(flat);
            let mut target = Array1::zeros(10);
            target[digit] = 1.0;
            labels.push(target);
            }
        }
    }
    let mut dataset = Dataset::new(images, labels);
    dataset = dataset.split(0.1).0; // Only use 10% for speed
    let (train_dataset, test_dataset) = dataset.split(0.8);

    let mut ffn = FeedForward::new(
        InputLayer::new(28 * 28),
        vec![
            HiddenLayer::new(128, Box::new(Sigmoid)),
            HiddenLayer::new(64, Box::new(Sigmoid)),
        ],
        OutputLayer::new(10, Box::new(Softmax)),
        Box::new(CrossEntropy),
    );

    ffn.set_batch_size(32);
    ffn.set_epochs(5);
    ffn.set_learning_rate(0.01);
    ffn.set_randseed(0);
    ffn.set_verbose(true);
    ffn.train(&train_dataset);

    for layer in ffn.hidden_layers.iter() {
        println!("Layer weights: {:?}", layer.weights);
        println!("Layer biases: {:?}", layer.biases);
    }

    println!("Accuracy {}", ffn.accuracy(&test_dataset));
    println!("Mean Squared Error {}", ffn.mse(&test_dataset));
}