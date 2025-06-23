use crate::nn::Forward;
use crate::nn::activations::{Relu, SoftMax};
use crate::nn::nn::{DenseLayer, Neuron};
use crate::nn::primitives::FPrimitive;
use rand::distr::StandardUniform;

pub struct Mnist<T> {
    l0: DenseLayer<T>,
    l1: DenseLayer<T>,
    l2: DenseLayer<T>,
    l3: DenseLayer<T>,
}

impl<T> Mnist<T>
where
    T: FPrimitive<T>
        + std::ops::Mul<T, Output = T>
        + std::ops::Add<T, Output = T>
        + std::ops::Div<T, Output = T>
        + Clone
        + Copy
        + Send
        + Sync
        + 'static
        + std::cmp::PartialOrd<f32>,
    StandardUniform: rand::distr::Distribution<T>,
{
    pub fn new() -> Mnist<T> {
        let mut l0: DenseLayer<T> = DenseLayer::new("l0", 784, vec![28, 28], false);
        let mut l1: DenseLayer<T> = DenseLayer::new("l1", 256, vec![784, 1], true);
        let mut l2: DenseLayer<T> = DenseLayer::new("l2", 128, vec![256, 1], true);
        let mut l3: DenseLayer<T> = DenseLayer::new("l3", 10, vec![128, 1], false);

        let mut n_count = 0;
        for layer in vec![&mut l0, &mut l1, &mut l2, &mut l3] {
            for neuron in layer.neurons.iter_mut() {
                n_count += neuron.params();
            }
        }

        println!("Parameter Count: {}", n_count);

        Mnist { l0, l1, l2, l3 }
    }

    pub async fn forward(&mut self, input: &Vec<T>) -> Vec<T> {
        let relu = Relu::new();
        let smax: SoftMax = SoftMax::new();

        let mut s = self.l0.forward(input.clone()).await;
        s = relu.forward(&s);
        s = self.l1.forward(s).await;
        s = relu.forward(&s);
        s = self.l2.forward(s).await;
        s = relu.forward(&s);
        s = self.l3.forward(s).await;
        s = smax.forward(&s);

        s
    }
}
