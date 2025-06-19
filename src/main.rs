pub mod types;

use types::primitives::FPrimitive as _;

use crate::types::primitives::{_F8, _F16, _F32};
use crate::types::{Layer, Neuron};

fn main() {
    let mut l1: Layer<_F8> = Layer::new(12, vec![1, 3]);
    let mut n1: Neuron<_F8> = Neuron::new(vec![1, 3]);

    let input: Vec<_F8> = vec![_F8::new(1.0f32), _F8::new(2.0f32), _F8::new(3.03f32)];
    println!("n1.sum(input): {:?}", n1.sum(input));
}
