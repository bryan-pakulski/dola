use std::cmp::Ordering;

use float8::F8E4M3;
use float16::f16;

use rand::Rng;
use rand::distr::{Distribution, StandardUniform};

// TODO: value should be of the base type i.e f32, f16, f8
// TODO: this should also be true of the new function which can take either f32, f16, f8
pub trait FPrimitive<T> {
    fn new(val: T) -> Self;
    fn default() -> Self;
    fn value(&self) -> T;
    fn max(&self, val: T) -> T;
}

#[derive(Debug, Clone, Copy)]
pub struct _F32 {
    pub val: f32,
}
#[derive(Debug, Clone, Copy)]
pub struct _F16 {
    pub val: f16,
}
#[derive(Debug, Clone, Copy)]
pub struct _F8 {
    pub val: F8E4M3,
}

// =================== FLOAT 32 IMPLEMENTATION ===================
impl FPrimitive<f32> for _F32 {
    fn new(val: f32) -> Self {
        _F32 { val }
    }
    fn default() -> Self {
        _F32::new(0.0f32)
    }
    fn value(&self) -> f32 {
        self.val
    }
    fn max(&self, val: f32) -> f32 {
        if self.val > val { self.val } else { val }
    }
}

impl FPrimitive<_F32> for _F32 {
    fn new(val: _F32) -> Self {
        _F32 { val: val.val }
    }
    fn default() -> Self {
        _F32::new(0.0f32)
    }
    fn value(&self) -> _F32 {
        _F32 { val: self.val }
    }
    fn max(&self, val: _F32) -> _F32 {
        if self.val > val.val { *self } else { val }
    }
}

impl FPrimitive<f16> for _F32 {
    fn new(val: f16) -> Self {
        _F32 { val: val.to_f32() }
    }
    fn default() -> Self {
        _F32::new(0.0f32)
    }
    fn value(&self) -> f16 {
        f16::from_f32(self.val)
    }
    fn max(&self, val: f16) -> f16 {
        if self.val > val.to_f32() {
            f16::from_f32(self.val)
        } else {
            val
        }
    }
}

impl FPrimitive<F8E4M3> for _F32 {
    fn new(val: F8E4M3) -> Self {
        _F32 { val: val.to_f32() }
    }
    fn default() -> Self {
        _F32::new(0.0f32)
    }
    fn value(&self) -> F8E4M3 {
        float8::F8E4M3::from_f32(self.val)
    }
    fn max(&self, val: F8E4M3) -> F8E4M3 {
        if self.val > val.to_f32() {
            float8::F8E4M3::from_f32(self.val)
        } else {
            val
        }
    }
}

impl From<_F32> for _F16 {
    fn from(f: _F32) -> Self {
        _F16 {
            val: f16::from_f32(f.val),
        }
    }
}

impl From<_F32> for _F8 {
    fn from(f: _F32) -> Self {
        _F8 {
            val: float8::F8E4M3::from_f32(f.val),
        }
    }
}

impl PartialEq for _F32 {
    fn eq(&self, other: &_F32) -> bool {
        self.val == other.val
    }
}

impl PartialEq<f32> for _F32 {
    fn eq(&self, other: &f32) -> bool {
        self.val == *other
    }
}

impl PartialOrd<f32> for _F32 {
    fn partial_cmp(&self, other: &f32) -> Option<Ordering> {
        self.val.partial_cmp(other)
    }
}

impl PartialOrd for _F32 {
    fn partial_cmp(&self, other: &_F32) -> Option<Ordering> {
        self.val.partial_cmp(&other.val)
    }
}

// =================== FLOAT 16 IMPLEMENTATION ===================

impl FPrimitive<f32> for _F16 {
    fn new(val: f32) -> Self {
        _F16 {
            val: f16::from_f32(val),
        }
    }
    fn default() -> Self {
        _F16::new(0.0f32)
    }
    fn value(&self) -> f32 {
        self.val.to_f32()
    }
    fn max(&self, val: f32) -> f32 {
        if self.val.to_f32() > val {
            self.val.to_f32()
        } else {
            val
        }
    }
}

impl FPrimitive<f16> for _F16 {
    fn new(val: f16) -> Self {
        _F16 { val }
    }
    fn default() -> Self {
        _F16::new(0.0f32)
    }
    fn value(&self) -> f16 {
        self.val
    }
    fn max(&self, val: f16) -> f16 {
        if self.val > val { self.val } else { val }
    }
}

impl FPrimitive<_F16> for _F16 {
    fn new(val: _F16) -> Self {
        _F16 { val: val.val }
    }
    fn default() -> Self {
        _F16::new(0.0f32)
    }
    fn value(&self) -> _F16 {
        _F16 { val: self.val }
    }
    fn max(&self, val: _F16) -> _F16 {
        if self.val > val.val { *self } else { val }
    }
}

impl FPrimitive<F8E4M3> for _F16 {
    fn new(val: F8E4M3) -> Self {
        _F16 {
            val: f16::from_f32(val.to_f32()),
        }
    }
    fn default() -> Self {
        _F16::new(0.0f32)
    }
    fn value(&self) -> F8E4M3 {
        float8::F8E4M3::from_f32(self.val.to_f32())
    }
    fn max(&self, val: F8E4M3) -> F8E4M3 {
        if self.val.to_f32() > val.to_f32() {
            float8::F8E4M3::from_f32(self.val.to_f32())
        } else {
            val
        }
    }
}

impl From<_F16> for _F32 {
    fn from(f: _F16) -> Self {
        _F32 {
            val: f.val.to_f32(),
        }
    }
}

impl From<_F16> for _F8 {
    fn from(f: _F16) -> Self {
        _F8 {
            val: float8::F8E4M3::from_f32(f.val.to_f32()),
        }
    }
}

impl PartialEq for _F16 {
    fn eq(&self, other: &_F16) -> bool {
        self.val == other.val
    }
}

impl PartialEq<f32> for _F16 {
    fn eq(&self, other: &f32) -> bool {
        self.val == f16::from_f32(*other)
    }
}

impl PartialOrd<f32> for _F16 {
    fn partial_cmp(&self, other: &f32) -> Option<Ordering> {
        self.val.partial_cmp(&f16::from_f32(*other))
    }
}

impl PartialOrd for _F16 {
    fn partial_cmp(&self, other: &_F16) -> Option<Ordering> {
        self.val.partial_cmp(&other.val)
    }
}

// =================== FLOAT 8 IMPLEMENTATION ===================

impl FPrimitive<f32> for _F8 {
    fn new(val: f32) -> Self {
        _F8 {
            val: float8::F8E4M3::from_f32(val),
        }
    }
    fn default() -> Self {
        _F8::new(0.0f32)
    }
    fn value(&self) -> f32 {
        self.val.to_f32()
    }
    fn max(&self, val: f32) -> f32 {
        if self.val.to_f32() > val {
            self.val.to_f32()
        } else {
            val
        }
    }
}

impl FPrimitive<f16> for _F8 {
    fn new(val: f16) -> Self {
        _F8 {
            val: float8::F8E4M3::from_f32(val.to_f32()),
        }
    }
    fn default() -> Self {
        _F8::new(0.0f32)
    }
    fn value(&self) -> f16 {
        f16::from_f32(self.val.to_f32())
    }
    fn max(&self, val: f16) -> f16 {
        if self.val.to_f32() > val.to_f32() {
            f16::from_f32(self.val.to_f32())
        } else {
            val
        }
    }
}

impl FPrimitive<F8E4M3> for _F8 {
    fn new(val: F8E4M3) -> Self {
        _F8 { val }
    }
    fn default() -> Self {
        _F8::new(0.0f32)
    }
    fn value(&self) -> F8E4M3 {
        self.val
    }
    fn max(&self, val: F8E4M3) -> F8E4M3 {
        if self.val > val { self.val } else { val }
    }
}

impl FPrimitive<_F8> for _F8 {
    fn new(val: _F8) -> Self {
        _F8 { val: val.val }
    }
    fn default() -> Self {
        _F8::new(0.0f32)
    }
    fn value(&self) -> _F8 {
        _F8 { val: self.val }
    }
    fn max(&self, val: _F8) -> _F8 {
        if self.val > val.val { *self } else { val }
    }
}

impl From<_F8> for _F32 {
    fn from(f: _F8) -> Self {
        _F32 {
            val: f.val.to_f32(),
        }
    }
}

impl From<_F8> for _F16 {
    fn from(f: _F8) -> Self {
        _F16 {
            val: f16::from_f32(f.val.to_f32()),
        }
    }
}

impl PartialEq for _F8 {
    fn eq(&self, other: &_F8) -> bool {
        self.val == other.val
    }
}

impl PartialEq<f32> for _F8 {
    fn eq(&self, other: &f32) -> bool {
        self.val == float8::F8E4M3::from_f32(*other)
    }
}

impl PartialOrd<f32> for _F8 {
    fn partial_cmp(&self, other: &f32) -> Option<Ordering> {
        self.val.partial_cmp(&float8::F8E4M3::from_f32(*other))
    }
}

impl PartialOrd for _F8 {
    fn partial_cmp(&self, other: &_F8) -> Option<Ordering> {
        self.val.partial_cmp(&other.val)
    }
}

// =================== RAND Distribution IMPLEMENTATION ===================
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
