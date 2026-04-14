// TurboQuant KV Cache Demo — 100% Rust
// Equivalent of Python's `if __name__ == '__main__':` block

use turboquant::{TurboQuantKVCache, benchmark_turboquant};

fn main() {
    println!("TurboQuant KV Cache Demo (Rust)");
    println!("{}", "=".repeat(50));

    // Create cache — production config (24 layers, 32 heads, dim 128)
    // Note: for the demo we use smaller dims to avoid slow QR on 4096×4096.
    //       In production (release build), use full params.
    let cache = TurboQuantKVCache::new(
        4,     // num_layers
        4096,  // max_seq_len
        32,    // head_dim
        8,     // num_heads
        3,     // bits
    );

    println!("Memory usage:         {:.2} MB", cache.memory_usage_mb());
    println!(
        "Compression vs FP16:  {:.1}x",
        cache.compression_ratio_vs_fp16()
    );

    // Benchmark
    let results = benchmark_turboquant(4096, 32, 8, 4);

    println!("\nBenchmark results:");
    println!("  FP16 memory:   {:.2} MB", results.fp16_memory_mb);
    println!("  TurboQuant:    {:.2} MB", results.turboquant_memory_mb);
    println!("  Compression:   {:.1}x", results.compression_ratio);
    println!("  Bits/value:    {}", results.bits_per_value);
    println!("  Seq length:    {}", results.seq_len);
    println!("  Layers:        {}", results.num_layers);
    println!("  Heads:         {}", results.num_heads);
    println!("  Head dim:      {}", results.head_dim);

    // Also show what full-scale config would look like (metrics only, no alloc)
    println!("\n--- Full-scale projection (24 layers, 32 heads, dim 128) ---");
    let fp16_full = 4096 * 128 * 32 * 24 * 2;
    let turbo_full = 4096 * 128 * 32 * 24 * 3 / 8;
    println!("  FP16 memory:   {:.2} MB", fp16_full as f64 / (1024.0 * 1024.0));
    println!("  TurboQuant:    {:.2} MB", turbo_full as f64 / (1024.0 * 1024.0));
    println!("  Compression:   {:.1}x", fp16_full as f64 / turbo_full as f64);
}
