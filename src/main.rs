pub mod nets;
pub mod nn;

use nets::calc::Calculator;
use nn::loss::{LossForward, MeanSquaredError};
use nn::primitives::{_F8, _F16, _F32, FPrimitive};
use rand::Rng;

fn main() {
    let mut train_dataset: Vec<(Vec<_F32>, Vec<_F32>)> = Vec::new();
    let mut rng = rand::rng();

    for _ in 0..100 {
        // Generate random numbers to add and the expected output
        let a: i32 = rng.random_range(0..10000);
        let b: i32 = rng.random_range(0..10000);
        let c: i32 = a + b;

        let input: Vec<_F32> = vec![_F32::new(a as f32), _F32::new(b as f32)];
        let output: Vec<_F32> = vec![_F32::new(c as f32)];

        train_dataset.push((input, output));
    }

    let mut cnet: Calculator<_F32>;
    cnet = Calculator::new();
    let loss_fn = MeanSquaredError {};

    for i in 0..1000 {
        //println!("Epoch {}", i);

        for (input, output) in train_dataset.iter() {
            let prediction: Vec<_F32> = cnet.forward(input);
            let loss_value: f32 = loss_fn.forward(&prediction, output);

            // Compute loss
            // println!("Prediction: {:?}", prediction);
            // println!("Output: {:?}", output);
            // println!("Loss: {}", loss_value);
        }
    }
}
