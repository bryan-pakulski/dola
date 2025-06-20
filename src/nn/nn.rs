use crate::nn::primitives::FPrimitive;
use rand::{distr::StandardUniform, prelude::*};

#[derive(Debug, Clone)]
pub struct Neuron<T> {
    weights: Vec<T>,
    bias: T,
}

#[derive(Debug, Clone)]
pub struct DenseLayer<T> {
    layer_name: String,
    input_dim: Vec<usize>,
    neurons: Vec<Neuron<T>>,
}

impl<T> Neuron<T>
where
    T: FPrimitive<T> + std::ops::Mul<T, Output = T> + std::ops::Add<T, Output = T> + Clone,
    StandardUniform: rand::distr::Distribution<T>,
{
    pub fn new(inputs: usize) -> Neuron<T> {
        let mut rng = rand::rng();

        let bias = T::from(rng.random::<T>()); // Initialize bias as f32 and convert to T
        let weights: Vec<T> = (0..inputs).map(|_| T::from(rng.random::<T>())).collect();

        Neuron { weights, bias }
    }

    pub fn sum(&self, inputs: &Vec<T>) -> T {
        assert_eq!(
            inputs.len(),
            self.weights.len(),
            "Neuron received invalid input shape!",
        );

        let mut output = T::default();

        for (idx, value) in inputs.iter().enumerate() {
            output = output.value() + value.value() * self.weights[idx].value();
        }

        T::new(output.value() + self.bias.value())
    }
}

impl<T> DenseLayer<T>
where
    T: FPrimitive<T> + std::ops::Mul<T, Output = T> + std::ops::Add<T, Output = T> + Clone,
    StandardUniform: rand::distr::Distribution<T>,
{
    pub fn new(layer_name: &str, neurons: usize, input_dim: Vec<usize>) -> DenseLayer<T> {
        let size = input_dim.iter().copied().reduce(|a, b| a * b).unwrap();

        let mut l: DenseLayer<T> = DenseLayer {
            layer_name: layer_name.into(),
            input_dim: input_dim.clone(),
            neurons: vec![],
        };

        for _ in 0..neurons {
            l.neurons.push(Neuron::new(size));
        }
        l
    }

    pub fn forward(&mut self, input: Vec<T>) -> Vec<T> {
        let size = self.input_dim.iter().copied().reduce(|a, b| a * b).unwrap();
        if size != input.len() {
            panic!(
                "Layer {} received invalid input shape! Expected {:?}",
                self.layer_name, self.input_dim
            );
        }

        let mut output = Vec::new();
        for neuron in self.neurons.iter() {
            output.push(neuron.sum(&input));
        }

        output
    }
}
