// Link against librknnrt: third_party/ → lib/ → RKNNRT_LIB_DIR env → system path.
fn main() {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").ok();
    let lib_dir = std::env::var("RKNNRT_LIB_DIR").ok().or_else(|| {
        manifest.as_ref().and_then(|m| {
            let base = std::path::Path::new(m);
            for subdir in ["third_party", "lib"] {
                let path = base.join(subdir);
                if path.join("librknnrt.so").exists() {
                    return path.canonicalize().ok().map(|p| p.to_string_lossy().into_owned());
                }
            }
            None
        })
    });
    if let Some(dir) = lib_dir {
        println!("cargo:rustc-link-search=native={}", dir);
        // target/release/dreamshaper-cli → $ORIGIN/../../third_party (local dev)
        println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN/../../third_party");
        // embed build-time path (cargo install / Docker builder)
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", dir);
    }
}
