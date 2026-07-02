//! Best-effort C smoke test: compiles a small C program against
//! `include/turboquant.h` and the freshly built static library, then runs
//! it. Skips gracefully (with a message) when no C compiler or static
//! library is available.

use std::path::PathBuf;
use std::process::Command;

/// C program exercising the full ABI: version, sizing, create/destroy,
/// quantize/dequantize round-trip, and a couple of error paths.
const C_SOURCE: &str = r#"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "turboquant.h"

int main(void) {
    const char *version = tq_version();
    if (version == NULL || strlen(version) == 0) return 1;

    tq_quantizer *q = NULL;
    if (tq_quantizer_create(3, 64, TQ_SCALE_ABSMAX, 0.0f, 1, 0.01f, &q) != TQ_OK) return 2;
    if (q == NULL) return 3;

    enum { N = 64 };
    float input[N];
    for (int i = 0; i < N; i++) input[i] = (float)(i - 32) / 10.0f;

    size_t packed_cap = tq_packed_size(N);
    size_t corr_cap = tq_corr_size(N);
    if (packed_cap != 24 || corr_cap != 8) return 4;

    uint8_t *packed = (uint8_t *)malloc(packed_cap);
    uint8_t *corr = (uint8_t *)malloc(corr_cap);
    if (packed == NULL || corr == NULL) return 5;

    size_t packed_written = 0, corr_written = 0;
    float scale = 0.0f;
    int rc = tq_quantize(q, input, N, packed, packed_cap, &packed_written,
                         &scale, corr, corr_cap, &corr_written);
    if (rc != TQ_OK) return 6;
    if (packed_written != packed_cap || corr_written != corr_cap) return 7;
    if (!(scale > 0.0f)) return 8;

    float output[N];
    rc = tq_dequantize(q, packed, packed_written, N, scale,
                       corr, corr_written, output, N);
    if (rc != TQ_OK) return 9;

    double mse = 0.0;
    for (int i = 0; i < N; i++) {
        double d = (double)input[i] - (double)output[i];
        mse += d * d;
    }
    mse /= N;
    if (mse > 0.25) {
        fprintf(stderr, "mse too high: %f\n", mse);
        return 10;
    }

    /* Error paths. */
    if (tq_quantize(NULL, input, N, packed, packed_cap, &packed_written,
                    &scale, corr, corr_cap, &corr_written) != TQ_ERR_NULL_POINTER) return 11;
    if (tq_quantize(q, input, N, packed, packed_cap - 1, &packed_written,
                    &scale, corr, corr_cap, &corr_written) != TQ_ERR_BUFFER_TOO_SMALL) return 12;
    if (tq_quantize(q, input, 0, packed, packed_cap, &packed_written,
                    &scale, corr, corr_cap, &corr_written) != TQ_ERR_INVALID_ARGUMENT) return 13;
    if (tq_dequantize(q, packed, packed_cap, N, -1.0f,
                      corr, corr_cap, output, N) != TQ_ERR_INVALID_ARGUMENT) return 14;

    free(packed);
    free(corr);
    tq_quantizer_destroy(q);
    tq_quantizer_destroy(NULL);

    printf("c smoke ok: version=%s mse=%f\n", version, mse);
    return 0;
}
"#;

/// Directory holding the built library artifacts (`target/<profile>`),
/// derived from the test executable's own location
/// (`target/<profile>/deps/c_smoke-<hash>`).
fn artifact_dir() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    Some(exe.parent()?.parent()?.to_path_buf())
}

#[test]
fn c_smoke() {
    // Skip when no C compiler is available.
    if Command::new("cc").arg("--version").output().is_err() {
        eprintln!("skipping c_smoke: no `cc` on PATH");
        return;
    }

    let Some(lib_dir) = artifact_dir() else {
        eprintln!("skipping c_smoke: cannot locate target directory");
        return;
    };
    let static_lib = lib_dir.join("libturboquant_ffi.a");
    if !static_lib.exists() {
        eprintln!(
            "skipping c_smoke: {} not built (run `cargo build -p turboquant-ffi` first)",
            static_lib.display()
        );
        return;
    }

    let include_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("include");
    assert!(
        include_dir.join("turboquant.h").exists(),
        "include/turboquant.h is missing"
    );

    let work = tempfile::tempdir().expect("create temp dir");
    let c_file = work.path().join("smoke.c");
    let bin = work.path().join("smoke");
    std::fs::write(&c_file, C_SOURCE).expect("write smoke.c");

    // Compile with warnings-as-errors so the header itself must be clean C99.
    let compile = Command::new("cc")
        .arg("-std=c99")
        .arg("-Wall")
        .arg("-Wextra")
        .arg("-Werror")
        .arg("-I")
        .arg(&include_dir)
        .arg(&c_file)
        .arg("-L")
        .arg(&lib_dir)
        .arg("-lturboquant_ffi")
        .arg("-lpthread")
        .arg("-ldl")
        .arg("-lm")
        .arg("-o")
        .arg(&bin)
        .output()
        .expect("run cc");
    assert!(
        compile.status.success(),
        "C compilation failed:\n{}",
        String::from_utf8_lossy(&compile.stderr)
    );

    let run = Command::new(&bin).output().expect("run smoke binary");
    assert!(
        run.status.success(),
        "C smoke binary failed (status {:?}):\nstdout: {}\nstderr: {}",
        run.status.code(),
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr)
    );
    let stdout = String::from_utf8_lossy(&run.stdout);
    assert!(stdout.contains("c smoke ok"), "unexpected output: {stdout}");
}
