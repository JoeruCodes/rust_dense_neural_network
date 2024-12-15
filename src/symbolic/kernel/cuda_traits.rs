use std::ffi::c_int;

use num_traits::Float;

use super::{
    vector_add_f32, vector_add_f64, vector_dot_f32, vector_dot_f64, vector_mul_f32, vector_mul_f64,
};

pub trait SupportedFloat: Float {
    type CType;

    fn as_ptr(&self) -> *const Self::CType;
    fn as_mut_ptr(&mut self) -> *mut Self::CType;
}

impl SupportedFloat for f32 {
    type CType = f32;

    fn as_ptr(&self) -> *const Self::CType {
        self as *const f32
    }

    fn as_mut_ptr(&mut self) -> *mut Self::CType {
        self as *mut f32
    }
}

impl SupportedFloat for f64 {
    type CType = f64;

    fn as_ptr(&self) -> *const Self::CType {
        self as *const f64
    }

    fn as_mut_ptr(&mut self) -> *mut Self::CType {
        self as *mut f64
    }
}

pub trait CudaVectorAdd {
    type CType;

    unsafe fn add(
        a: *const Self::CType,
        b: *const Self::CType,
        c: *mut Self::CType,
        num_elements: c_int,
    ) -> c_int;
    unsafe fn mul(
        a: *const Self::CType,
        b: *const Self::CType,
        c: *mut Self::CType,
        num_elements: c_int,
    ) -> c_int;
    unsafe fn dot(
        a: *const Self::CType,
        b: *const Self::CType,
        c: *mut Self::CType,
        num_elements: c_int,
    ) -> c_int;
}

/// Implement CudaVectorAdd for f32
impl CudaVectorAdd for f32 {
    type CType = f32;

    unsafe fn add(
        a: *const Self::CType,
        b: *const Self::CType,
        c: *mut Self::CType,
        num_elements: c_int,
    ) -> c_int {
        vector_add_f32(a, b, c, num_elements)
    }

    unsafe fn dot(
        a: *const Self::CType,
        b: *const Self::CType,
        c: *mut Self::CType,
        num_elements: c_int,
    ) -> c_int {
        vector_dot_f32(a, b, c, num_elements)
    }

    unsafe fn mul(
        a: *const Self::CType,
        b: *const Self::CType,
        c: *mut Self::CType,
        num_elements: c_int,
    ) -> c_int {
        vector_mul_f32(a, b, c, num_elements)
    }
}

impl CudaVectorAdd for f64 {
    type CType = f64;

    unsafe fn add(
        a: *const Self::CType,
        b: *const Self::CType,
        c: *mut Self::CType,
        num_elements: c_int,
    ) -> c_int {
        vector_add_f64(a, b, c, num_elements)
    }

    unsafe fn dot(
        a: *const Self::CType,
        b: *const Self::CType,
        c: *mut Self::CType,
        num_elements: c_int,
    ) -> c_int {
        vector_dot_f64(a, b, c, num_elements)
    }

    unsafe fn mul(
        a: *const Self::CType,
        b: *const Self::CType,
        c: *mut Self::CType,
        num_elements: c_int,
    ) -> c_int {
        vector_mul_f64(a, b, c, num_elements)
    }
}
