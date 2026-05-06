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
        "  FP16 memory (K+V):     {:>8.2} MB",
        results.fp16_total_mb.to_string().yellow()
    );
    println!(
        "  TurboQuant K cache:    {:>8.2} MB",
        results.compressed_k_mb.to_string().yellow()
    );
    println!(
        "  TurboQuant V cache:    {:>8.2} MB",
        results.compressed_v_mb.to_string().yellow()
    );
    println!(
        "  Compression ratio:     {:>8.2}x",
        results.compression_ratio.to_string().bright_green().bold()
    );
    println!(
        "  Throughput:            {:>8.2} GB/s",
        results.compression_throughput_gbps.to_string().yellow()
    );
    println!(
        "  Elapsed:               {:>8.2} ms",
        results.elapsed_ms.to_string().yellow()
    );

    if results.compression_ratio >= 5.0 {
        println!(
            "\n{} Target compression ratio achieved (≥ 5.0×)",
            "✓".green().bold()
        );
    } else {
        println!(
            "\n{} Compression ratio below target ({} < 5.0×)",
            "⚠".yellow().bold(),
            format!("{:.1}", results.compression_ratio).yellow()
        );
    }

    Ok(())
}
