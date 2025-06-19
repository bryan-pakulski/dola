use float8::F8E4M3;
use float16::f16;

use rand::Rng;
use rand::distr::{Distribution, StandardUniform};


pub trait Primitive<T> {
    fn new(val: f32) -> Self;
    fn value(&self) -> &T;
    fn default() -> Self;
}

#[derive(Debug)]
pub struct _F32 {
    pub val: f32,
}
#[derive(Debug)]
pub struct _F16 {
    pub val: f16,
}
#[derive(Debug)]
pub struct _F8 {
    pub val: F8E4M3
}

impl Primitive<f32> for _F32 {
    fn new(val: f32) -> Self {
        _F32 { val }
    }
    fn value(&self) -> &f32 {
        &self.val
    }
    fn default() -> _F32 {
        _F32::new(0.0f32)
    }
}

impl Primitive<f16> for _F16 {
    fn new(val: f32) -> Self {
        _F16 { val: f16::from_f32(val) }
    }
    fn value(&self) -> &f16 {
        &self.val
    }
    fn default() -> _F16 {
        _F16::new(0.0f32)
    }
}

impl Primitive<F8E4M3> for _F8 {
    fn new(val: f32) -> Self {
        _F8 { val: float8::F8E4M3::from_f32(val) }
    }
    fn value(&self) -> &F8E4M3 {
        &self.val
    }
    fn default() -> _F8 {
        _F8::new(0.0f32)
    }
}


impl _F32 {
    pub fn new(val: f32) -> _F32 {
        _F32{val}
    }

    pub fn value(&self) -> &f32 {
        &self.val
    }
}

impl _F16 {
    pub fn new(val: f32) -> _F16 {
        _F16{val:f16::from_f32(val)}
    }
    pub fn value(&self) -> &f16 {
        &self.val
    }
}

impl _F8 {
    pub fn new(val: f32) -> _F8 {
        _F8{val:float8::F8E4M3::from_f32(val)}
    }
    pub fn value(&self) -> &F8E4M3 {
        &self.val
    }
}

// Distribution for Random Number generation

impl Distribution<_F32> for StandardUniform {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> _F32 {
        _F32::new(rng.random::<f32>())
    }
}

impl Distribution<StandardUniform> for _F32 {
    fn sample<R: Rng + ?Sized>(&self, _rng: &mut R) -> StandardUniform {
        StandardUniform
    }
}

impl Distribution<_F16> for StandardUniform {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> _F16 {
        _F16::new(rng.random::<f32>())

    }
}

impl Distribution<StandardUniform> for _F16 {
    fn sample<R: Rng + ?Sized>(&self, _rng: &mut R) -> StandardUniform {
        StandardUniform
    }
}

impl Distribution<_F8> for StandardUniform {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> _F8 {
        _F8::new(rng.random::<f32>())
    }
}

impl Distribution<StandardUniform> for _F8 {
    fn sample<R: Rng + ?Sized>(&self, _rng: &mut R) -> StandardUniform {
        StandardUniform
    }
}
