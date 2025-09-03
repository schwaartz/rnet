# rnet

A minimal neural network library in Rust for learning and experimentation.

The library can make simple feedforward networks of any shape and with any set of activation functions. It does not use GPU acceleration, so it is quite slow. Nonetheless, it is able to train a semi-decent classifier for the MNIST digits dataset (see [`src/main.rs`](src/main.rs)).

**Example:**

```rust
let mut ffn = FeedForward::new(
    InputLayer::new(28 * 28),
    vec![
        HiddenLayer::new(128, Box::new(Sigmoid)),
        HiddenLayer::new(64, Box::new(Sigmoid)),
    ],
    OutputLayer::new(10, Box::new(Softmax)),
    Box::new(CrossEntropy),
);
ffn.train(&train_dataset);
let acc = rnet.accuracy(&test_dataset);
println!("Test accuracy: {}", acc);
```

MIT License
