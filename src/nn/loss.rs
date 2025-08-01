pub trait LossForward {
    fn forward(&self, input: &Vec<f32>, target: &Vec<f32>) -> f32;
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

impl LossForward for MeanSquaredError {
    fn forward(&self, input: &Vec<f32>, target: &Vec<f32>) -> f32 {
        let mut loss = 0.0f32;

        for (i, t) in input.iter().zip(target.iter()) {
            loss += -(t - i);
        }

        loss
    }
}
