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
    // let dataset = dataset.split(0.05).1; // remove 90%
    let (train_dataset, test_dataset) = dataset.split(0.8);

    let activations = vec![Activation::ReLu, Activation::ReLu, Activation::None]; // The last layer will get softmaxed anyways
    let mut rnet = RNet::new(vec![28*28, 128, 10], activations, UseCase::Classification);

    rnet.set_learning_rate(0.1);
    rnet.set_epochs(1);
    rnet.set_batch_size(32);
    rnet.train(&train_dataset);

    let save_file_path = format!("{}/saves/weights_1.txt", std::env::current_dir().unwrap().display());
    if let Some(parent) = std::path::Path::new(&save_file_path).parent() {
        if !parent.exists() {
            fs::create_dir_all(parent).unwrap();
        }
    }
    rnet.save_weights(&save_file_path).unwrap();

    let accuracy = rnet.accuracy(&test_dataset);
    println!("Test accuracy: {}", accuracy);
}
