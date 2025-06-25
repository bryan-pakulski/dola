use super::Forward;

pub struct Relu {}
pub struct SoftMax {}

impl Relu {
    pub fn new() -> Relu {
        Relu {}
    }
}

impl Forward for Relu {
    fn forward(&self, input: &Vec<f32>) -> Vec<f32> {
        let mut output: Vec<f32> = Vec::new();

        for i in input.iter() {
            if *i > 0.0f32 {
                output.push(*i);
            } else {
                output.push(0.0f32);
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

impl Forward for SoftMax {
    fn forward(&self, input: &Vec<f32>) -> Vec<f32> {
        let mut output: Vec<f32> = Vec::new();
        let mut sum = 0.0f32;
        for i in input {
            sum = sum + *i;
        }
        for i in input {
            output.push(i / sum);
        }

        output
    }
}
