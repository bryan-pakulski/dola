use rand::prelude::*;
use std::sync::Arc;

use rayon::prelude::*;

#[derive(Debug, Clone)]
pub struct Neuron {
    weights: Vec<f32>,
    bias: f32,
    with_bias: bool,
}

#[derive(Debug, Clone)]
pub struct DenseLayer {
    pub layer_name: String,
    pub input_dim: Vec<usize>,
    pub neurons: Vec<Arc<Neuron>>,
    pub freeze: bool,
}

impl Neuron {
    pub fn new(inputs: usize, with_bias: bool) -> Neuron {
        let mut rng = rand::rng();

        let bias = rng.random::<f32>();
        let weights: Vec<f32> = (0..inputs).map(|_| rng.random::<f32>()).collect();

        Neuron {
            weights,
            bias,
            with_bias,
        }
    }

    pub fn sum(&self, inputs: &Vec<f32>) -> f32 {
        assert_eq!(
            inputs.len(),
            self.weights.len(),
            "Neuron received invalid input shape!",
        );

        let mut output = 0.0f32;

        // TODO: this could potentially also be done in parallel
        for (idx, value) in inputs.iter().enumerate() {
            output = output + value * self.weights[idx]
        }

        if self.with_bias {
            output = output + self.bias;
        }

        output
    }

    pub fn params(&self) -> usize {
        self.weights.len() + 1
    }
}

impl DenseLayer {
    pub fn new(
        layer_name: &str,
        neurons: usize,
        input_dim: Vec<usize>,
        with_bias: bool,
    ) -> DenseLayer {
        let size = input_dim.iter().copied().reduce(|a, b| a * b).unwrap();
        let mut l = DenseLayer {
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

    pub fn forward(&self, input: Vec<f32>) -> Vec<f32> {
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
