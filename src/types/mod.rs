pub mod ops;
pub mod primitives;
use crate::types::primitives::{_F8, _F16, _F32, FPrimitive};
use rand::{distr::StandardUniform, prelude::*};

#[derive(Debug)]
pub struct Neuron<T> {
    dims: Vec<usize>,
    weights: Vec<T>,
    bias: T,
}

pub struct Layer<T> {
    neurons: Vec<Neuron<T>>,
}

impl<T> Neuron<T>
where
    T: FPrimitive<T> + std::ops::Mul<T, Output = T> + std::ops::Add<T, Output = T>,
    StandardUniform: rand::distr::Distribution<T>,
{
    pub fn new(dimensions: Vec<usize>) -> Neuron<T> {
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

        let mut output = T::default();

        for (idx, value) in inputs.iter().enumerate() {
            output = output.value() + value.value() * self.weights[idx].value();
        }

        T::new(output.value() + self.bias.value())
    }
}

impl<T> Layer<T>
where
    T: FPrimitive<T> + std::ops::Mul<T, Output = T> + std::ops::Add<T, Output = T>,
{
    pub fn new(neurons: usize, input_dim: Vec<usize>) -> Layer<T> {
        let n: Neuron<T> = Neuron::new(input_dim);

        Layer {
            neurons: vec![Neuron::new(input_dim); neurons],
        }
    }

    pub fn forward(&mut self, input: Vec<T>) -> Vec<T> {
        let mut output = Vec::new();

        for neuron in self.neurons.iter() {
            output.push(neuron.sum(input));
        }

        output
    }
}
