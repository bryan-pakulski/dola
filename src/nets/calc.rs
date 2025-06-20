use crate::nn::Forward;
use crate::nn::activations::Relu;
use crate::nn::nn::DenseLayer;
use crate::nn::primitives::FPrimitive;
use rand::distr::StandardUniform;

pub struct Calculator<T> {
    l0: DenseLayer<T>,
    l1: DenseLayer<T>,
    l2: DenseLayer<T>,
    l3: DenseLayer<T>,
}

impl<T> Calculator<T>
where
    T: FPrimitive<T>
        + std::ops::Mul<T, Output = T>
        + std::ops::Add<T, Output = T>
        + Clone
        + std::cmp::PartialOrd<f32>,
    StandardUniform: rand::distr::Distribution<T>,
{
    pub fn new() -> Calculator<T> {
        let mut l0: DenseLayer<T> = DenseLayer::new("l0", 25, vec![1, 2]);
        let mut l1: DenseLayer<T> = DenseLayer::new("l1", 25, vec![5, 5]);
        let mut l2: DenseLayer<T> = DenseLayer::new("l2", 25, vec![5, 5]);
        let mut l3: DenseLayer<T> = DenseLayer::new("l3", 1, vec![5, 5]);

        Calculator { l0, l1, l2, l3 }
    }

    pub fn forward(&mut self, input: &Vec<T>) -> Vec<T> {
        let relu = Relu::new();

        let mut s = self.l0.forward(input.clone());
        s = relu.forward(&s);
        s = self.l1.forward(s);
        s = relu.forward(&s);
        s = self.l2.forward(s);
        s = relu.forward(&s);
        s = self.l3.forward(s);
        s = relu.forward(&s);

        s
    }
}
