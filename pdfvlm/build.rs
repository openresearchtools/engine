fn main() {
    #[cfg(all(target_os = "windows", target_env = "msvc"))]
    {
        println!("cargo:rustc-link-lib=dylib=delayimp");
        println!("cargo:rustc-link-arg-cdylib=/DELAYLOAD:llama-server-bridge.dll");
    }
}
