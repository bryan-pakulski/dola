pub mod activations;
pub mod loss;
pub mod nn;
pub mod ops;
pub mod primitives;

use crate::nn::primitives::{_F8, _F16, _F32};

pub trait Forward<T> {
    fn forward(&self, input: &Vec<T>) -> Vec<T>;
}
