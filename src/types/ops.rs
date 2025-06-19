use std::ops::{Add, Sub, Mul};
use crate::types::{_F8, _F16, _F32};

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

impl Sub<_F32> for _F32 {
    type Output = Self;
    fn sub(self, other: _F32) -> _F32 {
        _F32::new(self.val - other.val)
    }
}
