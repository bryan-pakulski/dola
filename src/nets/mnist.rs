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
        + std::cmp::PartialOrd<f32>,
    StandardUniform: rand::distr::Distribution<T>,
{
    pub fn new() -> Mnist<T> {
        let mut l0: DenseLayer<T> = DenseLayer::new("l0", 256, vec![28, 28]);
        let mut l1: DenseLayer<T> = DenseLayer::new("l1", 30, vec![256, 1]);
        let mut l2: DenseLayer<T> = DenseLayer::new("l2", 30, vec![30, 1]);
        let mut l3: DenseLayer<T> = DenseLayer::new("l3", 10, vec![30, 1]);

        let mut n_count = 0;
        for layer in vec![&mut l0, &mut l1, &mut l2, &mut l3] {
            for neuron in layer.neurons.iter_mut() {
                n_count += neuron.params();
            }
        }

        println!("Parameter Count: {}", n_count);

        Mnist { l0, l1, l2, l3 }
    }

    pub fn forward(&mut self, input: &Vec<T>) -> Vec<T> {
        let relu = Relu::new();
        let smax: SoftMax = SoftMax::new();

        let mut s = self.l0.forward(input.clone());
        s = relu.forward(&s);
        s = self.l1.forward(s);
        s = relu.forward(&s);
        s = self.l2.forward(s);
        s = relu.forward(&s);
        s = self.l3.forward(s);
        s = smax.forward(&s);

        s
    }
}
