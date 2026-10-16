// The sherpa-onnx C library is found at run time where the build found it (the Python venv's copy, see build.sh):
// a dependency's own rpath setting doesn't reach this binary, so it is set here, and remembered for the sandbox.
fn main() {
    println!("cargo:rerun-if-env-changed=SHERPA_ONNX_LIB_DIR");
    if let Ok(dir) = std::env::var("SHERPA_ONNX_LIB_DIR") {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{dir}");
        println!("cargo:rustc-env=SHERPA_ONNX_LIB_DIR={dir}");
    }
}
