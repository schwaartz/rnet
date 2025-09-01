#![allow(dead_code)]

pub trait Activation {
    fn compute_value(&self, x: f32) -> f32;
    fn compute_derivative(&self, x: f32) -> f32;
}

pub struct ReLU;

impl Activation for ReLU {
    fn compute_value(&self, x: f32) -> f32 {
        x.max(0.0)
    }

    fn compute_derivative(&self, x: f32) -> f32 {
        if x > 0.0 { 1.0 } else { 0.0 }
    }
}

pub struct Sigmoid;

impl Activation for Sigmoid {
    fn compute_value(&self, x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    fn compute_derivative(&self, x: f32) -> f32 {
        let s = self.compute_value(x);
        s * (1.0 - s)
    }
}

pub struct Layer {
    weights: Vec<Vec<f32>>,
    bias: Vec<f32>,
    activation: Box<dyn Activation>, // trait object
}

impl Layer {
    pub fn new(weights: Vec<Vec<f32>>, bias: Vec<f32>, activation: Box<dyn Activation>) -> Self {
        Self { weights, bias, activation }
    }

    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        let mut output = vec![0.0; self.bias.len()];

        // simple linear transform: output = W * input + bias
        for (i, row) in self.weights.iter().enumerate() {
            let sum: f32 = row.iter()
                              .zip(input.iter())
                              .map(|(w, x)| w * x)
                              .sum();
            output[i] = self.activation.compute_value(sum + self.bias[i]);
        }

        output
    }
}

fn main() {
    let weights = vec![vec![0.5, -0.2], vec![0.1, 0.4]];
    let bias = vec![0.0, 0.0];

    let relu_layer = Layer::new(weights.clone(), bias.clone(), Box::new(ReLU));
    let sigmoid_layer = Layer::new(weights.clone(), bias.clone(), Box::new(Sigmoid));

    let input = vec![1.0, 2.0];

    println!("ReLU output: {:?}", relu_layer.forward(&input));
    println!("Sigmoid output: {:?}", sigmoid_layer.forward(&input));
}