use float8::F8E4M3;
use float16::f16;

use rand::Rng;
use rand::distr::{Distribution, StandardUniform};

#[derive(Debug)]
pub struct _F32(f32);
#[derive(Debug)]
pub struct _F16(f16);
#[derive(Debug)]
pub struct _F8(F8E4M3);

impl From<_F32> for f32 {
    fn from(f: _F32) -> Self {
        f.into()
    }
}

impl From<f32> for _F32 {
    fn from(f: f32) -> Self {
        _F32(f)
    }
}

impl From<_F16> for f32 {
    fn from(f: _F16) -> Self {
        f.into()
    }
}

impl From<f32> for _F16 {
    fn from(f: f32) -> Self {
        _F16(float16::f16::from_f32(f))
    }
}

impl From<_F32> for _F16 {
    fn from(f: _F32) -> Self {
        _F16(float16::f16::from_f32(f.into()))
    }
}

impl From<f32> for _F8 {
    fn from(f: f32) -> Self {
        _F8(float8::F8E4M3::from_f32(f))
    }
}

impl From<_F8> for f32 {
    fn from(f: _F8) -> Self {
        f.into()
    }
}

impl Distribution<_F32> for StandardUniform {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> _F32 {
        _F32(rng.random::<f32>())
    }
}

impl Distribution<_F16> for StandardUniform {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> _F16 {
        _F16(f16::from_f32(rng.random::<f32>()))
    }
}

impl Distribution<_F8> for StandardUniform {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> _F8 {
        _F8(rng.random::<f32>().into())
    }
}
