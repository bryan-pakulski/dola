pub mod types;

use crate::types::Neuron;
use crate::types::primitives::{_F16, _F32, _F8};

fn main() {
    let neuron: Neuron<_F32>;
    neuron = Neuron::init(vec![3, 2, 1]);

    print!("{:?}", neuron);
}
