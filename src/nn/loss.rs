use std::ops::{AddAssign, Neg};

use crate::nn::primitives::FPrimitive;

pub trait LossForward<T> {
    fn forward(&self, input: &Vec<T>, target: &Vec<T>) -> f32;
}

pub struct CategoriclalCrossEntropy {}
pub struct MeanSquaredError {}

// impl<T> LossForward<T> for CategoriclalCrossEntropy
// where
//     T: FPrimitive<T> + std::ops::Mul<T, Output = T> + std::ops::Add<T, Output = T> + Clone,
// {
//     fn forward(&self, input: &Vec<T>, target: &Vec<T>) -> f32 {
//         let mut loss = 0.0f32;
//
//         for (i, t) in input.iter().zip(target.iter()) {
//             loss += -(t.value() * (t.value().ln()));
//         }
//
//         loss
//     }
// }

impl<T> LossForward<T> for MeanSquaredError
where
    T: FPrimitive<T>
        + std::ops::Mul<T, Output = T>
        + std::ops::Add<T, Output = T>
        + std::ops::Sub<T, Output = T>
        + std::ops::Neg
        + Clone,
    f32: AddAssign<<T as Neg>::Output>,
{
    fn forward(&self, input: &Vec<T>, target: &Vec<T>) -> f32 {
        let mut loss = 0.0f32;

        for (i, t) in input.iter().zip(target.iter()) {
            loss += -(t.value() - i.value());
        }

        loss
    }
}
