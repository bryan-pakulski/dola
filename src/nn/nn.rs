use crate::nn::primitives::FPrimitive;
use rand::{distr::StandardUniform, prelude::*};
use std::ops::Deref;
use std::sync::Arc;
use tokio::sync::Mutex;
#[derive(Debug, Clone)]
pub struct Neuron<T> {
    weights: Vec<T>,
    bias: T,
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
    pub fn new(inputs: usize) -> Neuron<T> {
        let mut rng = rand::rng();

        let bias = T::from(rng.random::<T>()); // Initialize bias as f32 and convert to T
        let weights: Vec<T> = (0..inputs).map(|_| T::from(rng.random::<T>())).collect();

        Neuron { weights, bias }
    }

    async fn sum(&self, inputs: Arc<Vec<T>>) -> T {
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
        + Future
        + Send
        + Sync
        + 'static,
    StandardUniform: rand::distr::Distribution<T>,
{
    pub fn new(layer_name: &str, neurons: usize, input_dim: Vec<usize>) -> DenseLayer<T> {
        let size = input_dim.iter().copied().reduce(|a, b| a * b).unwrap();
        let mut l: DenseLayer<T> = DenseLayer {
            layer_name: layer_name.into(),
            input_dim: input_dim.clone(),
            neurons: vec![],
            freeze: false,
        };
        for _ in 0..neurons {
            l.neurons.push(Arc::new(Neuron::new(size)));
        }
        l
    }

    pub async fn forward(&self, input: Vec<T>) -> Vec<T> {
        let size = self.input_dim.iter().copied().reduce(|a, b| a * b).unwrap();
        if size != input.len() {
            panic!(
                "Layer {} received invalid input shape! Expected {:?} got {:?}",
                self.layer_name,
                self.input_dim,
                input.len()
            );
        }
        let shared_data = Arc::new(Mutex::new(Vec::new()));
        let shared_input = Arc::new(input);
        let mut handles = Vec::new();
        for neuron in self.neurons.iter() {
            let shared_data_clone = Arc::clone(&shared_data);
            let input_arc = Arc::clone(&shared_input);
            let neuron_clone = Arc::clone(neuron);
            let handle = tokio::spawn(async move {
                let nsum = neuron_clone.sum(input_arc).await;
                let mut data = shared_data_clone.lock().await;
                data.push(nsum);
            });
            handles.push(handle);
        }
        for handle in handles {
            handle.await.unwrap();
        }
        let data = shared_data.lock().await;
        data.deref().clone()
    }
}
