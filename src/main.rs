pub mod dataloader;
pub mod nets;
pub mod nn;
pub mod util;

use dataloader::classification::{ClassificationFolderLoader, Loader};
use nets::mnist::Mnist;
use nn::loss::{LossForward, MeanSquaredError};
use nn::primitives::{_F8, _F16, _F32, FPrimitive};

#[tokio::main]
async fn main() {
    let mut train_dataset: Loader<_F32> = Loader::new();
    train_dataset.load("/home/bryanp/dev/datasets/MNIST-JPG/train");

    println!("Train Dataset Size: {}", train_dataset.size());

    let mut cnet: Mnist<_F32>;
    cnet = Mnist::new();
    let loss_fn = MeanSquaredError {};

    for i in 0..100 {
        println!("Epoch {}", i);

        let mut sample: Option<_> = train_dataset.next();
        while sample.is_some() {
            match sample {
                Some((img_data, target)) => {
                    let prediction: Vec<_F32> = cnet.forward(&img_data).await;
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

            sample = train_dataset.next();
        }

        // Validation stage
    }
}
