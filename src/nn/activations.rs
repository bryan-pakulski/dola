use rand::distr::StandardUniform;

use super::Forward;
use super::primitives::FPrimitive;

pub struct Relu {}
pub struct SoftMax {}

impl Relu {
    pub fn new() -> Relu {
        Relu {}
    }
}

impl<T> Forward<T> for Relu
where
    T: FPrimitive<T>
        + std::ops::Mul<T, Output = T>
        + std::ops::Add<T, Output = T>
        + Clone
        + std::cmp::PartialOrd<f32>,
    StandardUniform: rand::distr::Distribution<T>,
{
    fn forward(&self, input: &Vec<T>) -> Vec<T> {
        let mut output: Vec<T> = Vec::new();

        for i in input.iter() {
            if i.value() > 0.0f32 {
                output.push(i.value());
            } else {
                output.push(T::default());
            }
        }

        output
    }
}

impl SoftMax {
    pub fn new() -> SoftMax {
        SoftMax {}
    }
}

// impl<T> Forward<T> for SoftMax
// where
//     T: FPrimitive<T> + std::ops::Mul<T, Output = T> + std::ops::Add<T, Output = T> + Clone,
//     StandardUniform: rand::distr::Distribution<T>,
// {
//     fn forward(&self, input: &Vec<T>) -> Vec<T> {
//         let mut output: Vec<T> = Vec::new();
//         let mut sum: T = T::default();
//         for i in input {
//             sum = sum + *i;
//         }
//         for i in input {
//             output.push(i.value() / sum);
//         }
//
//         output
//     }
// }
