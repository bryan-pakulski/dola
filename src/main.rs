pub mod dataloader;
pub mod nets;
pub mod nn;
pub mod util;

use dataloader::classification::{ClassificationFolderLoader, Loader};
use nets::mnist::Mnist;
use nn::loss::{LossForward, MeanSquaredError};
use nn::primitives::{_F8, _F16, _F32, FPrimitive};

use nn::nn::Neuron;

use indicatif::ProgressBar;
use std::sync::Arc;
use std::time::Duration;

//#[tokio::main]
//fn main() {
//    let n: Neuron<_F32> = Neuron::new(3, true);
//
//    println!("{:?}", n);
//
//    let input: Arc<Vec<_F32>> = Arc::new(vec![
//        _F32::new(1.0f32),
//        _F32::new(-2.0f32),
//        _F32::new(3.0f32),
//    ]);
//
//    let output = n.sum(input).await;
//
//    println!("{:?}", output);
//}

fn main() {
    let mut train_dataset: Loader<_F32> = Loader::new();
    train_dataset.load("/home/bryanp/dev/datasets/MNIST-JPG/test");

    println!("Train Dataset Size: {}", train_dataset.size());

    let mut cnet: Mnist<_F32>;
    cnet = Mnist::new();
    let loss_fn = MeanSquaredError {};

    for i in 0..100 {
        let start = std::time::Instant::now();
        let pb = ProgressBar::new(train_dataset.size() as u64);

        println!("Epoch {}", i);
        train_dataset.shuffle();

        let mut sample: Option<_> = train_dataset.next();
        while sample.is_some() {
            match sample {
                Some((img_data, target)) => {
                    let prediction: Vec<_F32> = cnet.forward(&img_data);
                    let loss_value: f32 = loss_fn.forward(&prediction, &target);
                    //println!("Prediction: {:?}", prediction);
                    //println!("Target: {:?}", target);
                    //println!("Loss: {}", loss_value);
                }
                None => {}
            }

            // Compute loss
            // println!("Prediction: {:?}", prediction);
            // println!("Output: {:?}", output);
            // println!("Loss: {}", loss_value);

            pb.inc(1);
            sample = train_dataset.next();
        }
        pb.finish_with_message(format!("Epoch {} took {}", i, start.elapsed().as_secs()));
        println!("Epoch {} finished in {}s", i, start.elapsed().as_secs());

        // Validation stage
    }
}
