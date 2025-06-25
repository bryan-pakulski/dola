pub mod activations;
pub mod loss;
pub mod nn;

pub trait Forward {
    fn forward(&self, input: &Vec<f32>) -> Vec<f32>;
}
