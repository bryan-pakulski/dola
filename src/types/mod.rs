pub mod primitives;
use rand::{distr::StandardUniform, prelude::*};

#[derive(Debug)]
pub struct Neuron<T> {
    dims: Vec<usize>,
    weights: Vec<T>,
    bias: T,
}

impl<T> Neuron<T>
where
    T: From<T> + rand::distr::Distribution<StandardUniform>,
    StandardUniform: rand::distr::Distribution<T>,
{
    pub fn init(dimensions: Vec<usize>) -> Neuron<T> {
        let mut rng = rand::rng();

        let bias = T::from(rng.random::<T>()); // Initialize bias as f32 and convert to T
        let size = dimensions.iter().copied().reduce(|a, b| a * b).unwrap();
        let weights: Vec<T> = (0..size).map(|_| T::from(rng.random::<T>())).collect();

        Neuron {
            dims: dimensions,
            weights,
            bias,
        }
    }
}
