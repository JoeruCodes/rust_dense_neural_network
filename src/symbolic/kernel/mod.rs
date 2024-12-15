use std::ffi::c_int;
pub mod cuda_traits;
#[link(name = "vector_add")]
extern "C" {
    fn vector_add_f32(A: *const f32, B: *const f32, C: *mut f32, num_elements: c_int) -> c_int;
    fn vector_add_f64(A: *const f64, B: *const f64, C: *mut f64, num_elements: c_int) -> c_int;
}

#[link(name = "vector_mul")]
extern "C" {
    fn vector_mul_f32(A: *const f32, B: *const f32, C: *mut f32, num_elements: c_int) -> c_int;

    fn vector_mul_f64(A: *const f64, B: *const f64, C: *mut f64, num_elements: c_int) -> c_int;
}

#[link(name = "vector_dot")]
extern "C" {
    fn vector_dot_f32(A: *const f32, B: *const f32, C: *mut f32, num_elements: c_int) -> c_int;
    fn vector_dot_f64(A: *const f64, B: *const f64, C: *mut f64, num_elements: c_int) -> c_int;
}
