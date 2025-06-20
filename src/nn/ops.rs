use crate::nn::{_F8, _F16, _F32};
use std::ops::{Add, AddAssign, Mul, Neg, Sub};

use super::primitives::FPrimitive;

impl Mul<_F32> for _F32 {
    type Output = Self;

    fn mul(self, other: _F32) -> _F32 {
        _F32::new(self.val * other.val)
    }
}

impl Add<_F32> for _F32 {
    type Output = Self;
    fn add(self, other: _F32) -> _F32 {
        _F32::new(self.val + other.val)
    }
}

impl AddAssign<_F32> for f32 {
    fn add_assign(&mut self, other: _F32) {
        *self += other.val;
    }
}

impl Sub<_F32> for _F32 {
    type Output = Self;
    fn sub(self, other: _F32) -> _F32 {
        _F32::new(self.val - other.val)
    }
}

impl Neg for _F32 {
    type Output = Self;
    fn neg(self) -> Self {
        _F32::new(-self.val)
    }
}

impl Mul<_F16> for _F16 {
    type Output = Self;

    fn mul(self, other: _F16) -> _F16 {
        _F16::new(self.val * other.val)
    }
}

impl Add<_F16> for _F16 {
    type Output = Self;
    fn add(self, other: _F16) -> _F16 {
        _F16::new(self.val + other.val)
    }
}

impl AddAssign<_F16> for f32 {
    fn add_assign(&mut self, other: _F16) {
        *self += other.val.as_f32();
    }
}

impl Sub<_F16> for _F16 {
    type Output = Self;
    fn sub(self, other: _F16) -> _F16 {
        _F16::new(self.val - other.val)
    }
}

impl Neg for _F16 {
    type Output = Self;
    fn neg(self) -> Self {
        _F16::new(-self.val)
    }
}

impl Mul<_F8> for _F8 {
    type Output = Self;

    fn mul(self, other: _F8) -> _F8 {
        _F8::new(self.val * other.val)
    }
}

impl Add<_F8> for _F8 {
    type Output = Self;
    fn add(self, other: _F8) -> _F8 {
        _F8::new(self.val + other.val)
    }
}

impl AddAssign<_F8> for f32 {
    fn add_assign(&mut self, other: _F8) {
        *self += other.val.to_f32();
    }
}

impl Sub<_F8> for _F8 {
    type Output = Self;
    fn sub(self, other: _F8) -> _F8 {
        _F8::new(self.val - other.val)
    }
}

impl Neg for _F8 {
    type Output = Self;
    fn neg(self) -> Self {
        _F8::new(-self.val)
    }
}
