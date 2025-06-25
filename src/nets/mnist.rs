use crate::nn::Forward;
use crate::nn::activations::{Relu, SoftMax};
use crate::nn::nn::DenseLayer;
use rand::distr::StandardUniform;

pub struct Mnist {
    l0: DenseLayer,
    l1: DenseLayer,
    l2: DenseLayer,
    l3: DenseLayer,
}

impl Mnist {
    pub fn new() -> Mnist {
        let mut l0: DenseLayer = DenseLayer::new("l0", 784, vec![28, 28], false);
        let mut l1: DenseLayer = DenseLayer::new("l1", 256, vec![784, 1], true);
        let mut l2: DenseLayer = DenseLayer::new("l2", 128, vec![256, 1], true);
        let mut l3: DenseLayer = DenseLayer::new("l3", 10, vec![128, 1], false);

        let mut params = 0;
        let mut neurons: usize = 0;

        for layer in vec![&mut l0, &mut l1, &mut l2, &mut l3] {
            for neuron in layer.neurons.iter_mut() {
                params += neuron.params();
                neurons += 1;
            }
        }

        println!("Neurons: {}", neurons);
        println!("Parameter Count: {}", params);

        Mnist { l0, l1, l2, l3 }
    }

    pub fn forward(&mut self, input: &Vec<f32>) -> Vec<f32> {
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
