//! Bench command implementation.

use colored::Colorize;
use turboquant_cpu::CpuBackend;

/// Run the bench command.
///
/// # Errors
///
/// Returns an error if benchmarking fails.
pub fn run(
    head_dim: usize,
    seq_len: usize,
    num_heads: usize,
    _iterations: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("{}", "TurboQuant Benchmark".bold().cyan());
    println!("{}", "═══════════════════".bold());
    println!();
    println!(
        "Configuration: head_dim={}, seq_len={}, num_heads={}",
        head_dim.to_string().cyan(),
        seq_len.to_string().cyan(),
        num_heads.to_string().cyan()
    );
    println!();

    let backend = CpuBackend::new();
    let results = backend.full_benchmark(head_dim, seq_len, num_heads);

    println!("{}", "Results:".bold().green());
    println!(
        "  FP16 memory (K+V):     {} MB",
        format!("{:>8.2}", results.fp16_total_mb).yellow()
    );
    println!(
        "  TurboQuant K cache:    {} MB",
        format!("{:>8.2}", results.compressed_k_mb).yellow()
    );
    println!(
        "  TurboQuant V cache:    {} MB",
        format!("{:>8.2}", results.compressed_v_mb).yellow()
    );
    println!(
        "  Compression ratio:     {}x",
        format!("{:>8.2}", results.compression_ratio)
            .bright_green()
            .bold()
    );
    println!(
        "  Throughput:            {} GB/s",
        format!("{:>8.2}", results.compression_throughput_gbps).yellow()
    );
    println!(
        "  Elapsed:               {} ms",
        format!("{:>8.2}", results.elapsed_ms).yellow()
    );

    // Theoretical ratio for 3-bit + 1-bit correction + per-block f16 scales
    // is 16 / 4.25 ≈ 3.76× (≈ 4.9× without correction) — flag only genuine
    // shortfalls against that, not against the correction-free ceiling.
    const TARGET_RATIO: f64 = 3.7;
    if results.compression_ratio >= TARGET_RATIO {
        println!(
            "\n{} Compression ratio matches the 3-bit + correction format (≥ {TARGET_RATIO}×)",
            "✓".green().bold()
        );
    } else {
        println!(
            "\n{} Compression ratio below the format's theoretical {TARGET_RATIO}× ({})",
            "⚠".yellow().bold(),
            format!("{:.1}", results.compression_ratio).yellow()
        );
    }

    Ok(())
}
