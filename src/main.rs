#![allow(dead_code)]

mod rnet;

use rnet::activation::*;
use rnet::data::Dataset;
use rnet::*;

use std::fs;
use image::ImageReader;
use ndarray::Array1;

fn main() {
    let mut images = Vec::new();
    let mut labels = Vec::new();

    for digit in 0..10 {
        let train_folder = format!("{}/data/mnist/dataset/{}", std::env::current_dir().unwrap().display(), digit);
        println!("Looking for images in: {}", train_folder);
        for entry in fs::read_dir(&train_folder).unwrap() {
            let entry = entry.unwrap();
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("jpg") {
            let img = ImageReader::open(&path).unwrap().decode().unwrap().to_luma8();
            let flat: Array1<f64> = Array1::from(
                img.pixels()
                .map(|p| p[0] as f64 / 255.0)
                .collect::<Vec<_>>()
            );
            images.push(flat);
            let mut target = Array1::zeros(10);
            target[digit] = 1.0;
            labels.push(target);
            }
        }
    }

    let dataset = Dataset::new(images, labels);
    let (train_dataset, test_dataset) = dataset.split(0.8);

    let mut rnet = RNet::new(vec![28*28, 128, 10], vec![Activation::Sigmoid, Activation::Sigmoid], UseCase::Classification);
    rnet.set_learning_rate(0.1);
    rnet.set_epochs(1);
    rnet.set_batch_size(32);
    rnet.train(&train_dataset);

    let accuracy = rnet.accuracy(&test_dataset);
    println!("Test accuracy: {}", accuracy);
}
