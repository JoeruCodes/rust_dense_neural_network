// build.rs

use std::env;

fn main() {
    println!("cargo:rerun-if-changed=src/vector_add.cu");
    println!("cargo:rerun-if-changed=src/vector_mul.cu");
    println!("cargo:rerun-if-changed=src/vector_dot.cu");

    // Compile the CUDA code
    cc::Build::new()
        .cuda(true)
        .flag("-cudart=shared")
        .flag("-arch=sm_61")
        .file("src/vector_add.cu")
        .compile("vector_add");

    cc::Build::new()
        .cuda(true)
        .flag("-cudart=shared")
        .flag("-arch=sm_61")
        .file("src/vector_mul.cu")
        .compile("vector_mul");

    cc::Build::new()
        .cuda(true)
        .flag("-cudart=shared")
        .flag("-arch=sm_61")
        .file("src/vector_dot.cu")
        .compile("vector_dot");

    let cuda_lib_path =
        env::var("CUDA_LIB_PATH").unwrap_or_else(|_| "/usr/local/cuda/lib64".to_string());

    println!("cargo:rustc-link-search=native={}", cuda_lib_path);
    println!("cargo:rustc-link-lib=cudart");

    println!(
        "cargo:rustc-link-search=native={}/lib64/stub",
        cuda_lib_path
    );
    println!("cargo:rustc-link-lib=cuda");
}
