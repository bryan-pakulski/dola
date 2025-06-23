use crate::nn::primitives::FPrimitive;
use rand::{distr::StandardUniform, prelude::*};
use std::ops::Deref;
use std::sync::Arc;

use rayon::prelude::*;

#[derive(Debug, Clone)]
pub struct Neuron<T> {
    weights: Vec<T>,
    bias: T,
    with_bias: bool,
}

#[derive(Debug, Clone)]
pub struct DenseLayer<T> {
    pub layer_name: String,
    pub input_dim: Vec<usize>,
    pub neurons: Vec<Arc<Neuron<T>>>,
    pub freeze: bool,
}

impl<T> Neuron<T>
where
    T: FPrimitive<T> + std::ops::Mul<T, Output = T> + std::ops::Add<T, Output = T> + Clone,
    StandardUniform: rand::distr::Distribution<T>,
{
    pub fn new(inputs: usize, with_bias: bool) -> Neuron<T> {
        let mut rng = rand::rng();

        let bias = T::from(rng.random::<T>());
        let weights: Vec<T> = (0..inputs).map(|_| T::from(rng.random::<T>())).collect();

        Neuron {
            weights,
            bias,
            with_bias,
        }
    }

    pub fn sum(&self, inputs: &Vec<T>) -> T {
        assert_eq!(
            inputs.len(),
            self.weights.len(),
            "Neuron received invalid input shape!",
        );

        let mut output = T::default();

        // TODO: this could potentially also be done in parallel
        for (idx, value) in inputs.iter().enumerate() {
            output = output.value() + value.value() * self.weights[idx].value();
        }

        if self.with_bias {
            output = output + self.bias.value();
        }

        output
    }

    pub fn params(&self) -> usize {
        self.weights.len() + 1
    }
}

impl<T> DenseLayer<T>
where
    T: FPrimitive<T>
        + std::ops::Mul<T, Output = T>
        + std::ops::Add<T, Output = T>
        + Clone
        + Send
        + Sync
        + 'static,
    StandardUniform: rand::distr::Distribution<T>,
{
    pub fn new(
        layer_name: &str,
        neurons: usize,
        input_dim: Vec<usize>,
        with_bias: bool,
    ) -> DenseLayer<T> {
        let size = input_dim.iter().copied().reduce(|a, b| a * b).unwrap();
        let mut l: DenseLayer<T> = DenseLayer {
            layer_name: layer_name.into(),
            input_dim: input_dim.clone(),
            neurons: vec![],
            freeze: false,
        };
        for _ in 0..neurons {
            l.neurons.push(Arc::new(Neuron::new(size, with_bias)));
        }
        l
    }

    pub fn forward(&self, input: Vec<T>) -> Vec<T> {
        let size = self.input_dim.iter().copied().reduce(|a, b| a * b).unwrap();
        if size != input.len() {
            panic!(
                "Layer {} received invalid input shape! Expected {:?} got {:?}",
                self.layer_name,
                self.input_dim,
                input.len()
            );
        }

        self.neurons
            .par_iter()
            .map(|neuron| {
                let nsum = neuron.sum(&input);
                nsum
            })
            .collect()
    }
}
