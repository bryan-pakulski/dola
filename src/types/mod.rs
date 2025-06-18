// The prelude import enables methods we use below, specifically
// Rng::random, Rng::sample, SliceRandom::shuffle and IndexedRandom::choose.
use rand::prelude::*;

use std::any::TypeId;
use {float16::f16, float8::F8E4M3};

pub struct Neuron<T> {
    inputs: Vec<T>,
    weights: Vec<T>,
    bias: T
}

enum Precision {
    F32(f32),
    F16(float16::f16),
    F8(float8::F8E4M3),
}

impl<T> Neuron<T> {
    fn init(&self, inputs: Vec<Precision>) -> Neuron<Precision> {
        let mut rng = rand::rng();
        let mut ne: Neuron<Precision>;
        for input in inputs {
            ne.inputs.push(input);

            match T {
                Precision::F32(T) => {
                    ne.weights.push(Precision::F32(rng.random::<f32>()));
                },
                Precision::F16 => {
                    ne.weights.push(rng.random::<float16>());
                },
                Precision::F8 => {
                    ne.weights.push(rng.random::<float8>());
                },
                _ => {
                    println!("Invalid type!")
                }
            }
            ne.weights.push(rng.random::<Precision>());
        }

        ne
    }
}

