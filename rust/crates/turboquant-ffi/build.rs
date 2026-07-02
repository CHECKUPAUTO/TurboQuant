//! Build script: regenerates `include/turboquant.h` from the Rust source
//! using cbindgen so the shipped C header always matches the compiled ABI.

use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=cbindgen.toml");

    let crate_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let header = crate_dir.join("include").join("turboquant.h");
    let config =
        cbindgen::Config::from_file(crate_dir.join("cbindgen.toml")).expect("read cbindgen.toml");

    match cbindgen::Builder::new()
        .with_crate(&crate_dir)
        .with_config(config)
        .generate()
    {
        Ok(bindings) => {
            bindings.write_to_file(&header);
        }
        Err(err) => {
            // Keep the committed header if regeneration fails (e.g. offline
            // tooling problems); fail hard only when no header exists at all.
            if header.exists() {
                println!(
                    "cargo:warning=cbindgen failed ({err}); keeping existing include/turboquant.h"
                );
            } else {
                panic!("cbindgen failed and include/turboquant.h does not exist: {err}");
            }
        }
    }
}
