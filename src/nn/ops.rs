use crate::nn::{_F8, _F16, _F32};
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub};

use super::primitives::FPrimitive;

impl Mul for _F32 {
    type Output = Self;

    fn mul(self, other: _F32) -> _F32 {
        _F32::new(self.val * other.val)
    }
}

impl Add for _F32 {
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

impl Sub for _F32 {
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

impl Div for _F32 {
    type Output = Self;
    fn div(self, other: _F32) -> _F32 {
        _F32::new(self.val / other.val)
    }
}

impl Mul for _F16 {
    type Output = Self;

    fn mul(self, other: _F16) -> _F16 {
        _F16::new(self.val * other.val)
    }
}

impl Add for _F16 {
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

impl Sub for _F16 {
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

impl Div for _F16 {
    type Output = Self;
    fn div(self, other: _F16) -> _F16 {
        _F16::new(self.val / other.val)
    }
}

impl Mul for _F8 {
    type Output = Self;

    fn mul(self, other: _F8) -> _F8 {
        _F8::new(self.val * other.val)
    }
}

impl Add for _F8 {
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

impl Sub for _F8 {
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

impl Div for _F8 {
    type Output = Self;
    fn div(self, other: _F8) -> _F8 {
        _F8::new(self.val / other.val)
    }
}
