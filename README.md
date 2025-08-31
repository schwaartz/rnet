# rnet

A minimal neural network library in Rust for learning and experimentation.

The library can make simple feedforward networks of any shape and with any set of activation functions. It does not use GPU acceleration, so it is quite slow. Nonetheless, it is able to train a semi-decent classifier for the MNIST digits dataset (see [`src/main.rs`](src/main.rs)).

**Example:**

```rust
let mut rnet = RNet::new(vec![784, 128, 10], vec![Activation::ReLu, Activation::ReLu, Activation::None], UseCase::Classification);
rnet.train(&train_dataset);
let acc = rnet.accuracy(&test_dataset);
println!("Test accuracy: {}", acc);
```

MIT License
