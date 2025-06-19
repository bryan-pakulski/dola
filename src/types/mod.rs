pub mod ops;
pub mod primitives;
use crate::types::primitives::{_F8, _F16, _F32};
use float8::F8E4M3;
use rand::{distr::StandardUniform, prelude::*};

use crate::types::primitives::Primitive;

#[derive(Debug)]
pub struct Neuron<T> {
    dims: Vec<usize>,
    weights: Vec<T>,
    bias: T,
}

impl<T> Neuron<T>
where
    T: Primitive<f32> + Primitive<float16::f16> + Primitive<F8E4M3>,
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

    pub fn sum(&mut self, inputs: Vec<T>) -> T {
        assert_eq!(
            inputs.len(),
            self.weights.len(),
            "Neuron received invalid input shape!"
        );

        let mut output = T::default(); // TODO: figure out how to initialize using `Default`

        for (idx, value) in inputs.iter().enumerate() {
            output = T::new(output.value() + value.value() * self.weights[idx].value());
        }

        T::new(output.value() + self.bias.value())
    }

    pub fn sub(&mut self, inputs: Vec<T>) -> T {


    }
}
