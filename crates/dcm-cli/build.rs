fn main() {
    // Upstream C++ parsing recurses deeply; bump the main-thread stack on macOS
    // for the optional `dcm2niix-ffi` parity binary only.
    if std::env::var("CARGO_FEATURE_FFI").is_ok()
        && std::env::var("CARGO_CFG_TARGET_OS").unwrap() == "macos"
    {
        println!("cargo:rustc-link-arg-bin=dcm2niix-ffi=-Wl,-stack_size,0x1000000");
    }
}
