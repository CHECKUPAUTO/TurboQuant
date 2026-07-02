#![deny(missing_docs)]
// Suppress pedantic clippy lints in CLI code
#![allow(clippy::unnecessary_wraps)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::items_after_statements)]
#![allow(clippy::needless_pass_by_value)]

//! `TurboQuant` CLI — 3-bit KV cache compression toolkit.
//!
//! Usage: `turboquant <COMMAND>`

use clap::{Parser, Subcommand, ValueHint};
use colored::Colorize;

mod commands;

/// `TurboQuant` 1.0.0 — 3-bit KV cache compression toolkit
#[derive(Parser)]
#[command(name = "turboquant")]
#[command(version = "1.0.0-rc1")]
#[command(about = "TurboQuant 3-bit KV cache compression toolkit", long_about = None)]
#[command(styles = clap_style())]
pub struct Cli {
    /// Subcommand to execute.
    #[command(subcommand)]
    pub command: Commands,

    /// Increase verbosity (-v, -vv, -vvv)
    #[arg(short = 'v', long = "verbose", action = clap::ArgAction::Count, global = true)]
    pub verbose: u8,

    /// Number of CPU threads (default: number of cores)
    #[arg(short = 'j', long = "threads", global = true)]
    pub threads: Option<usize>,

    /// Backend to use: cpu or auto (cuda is not implemented yet)
    #[arg(long = "backend", default_value = "auto", global = true)]
    pub backend: String,

    /// Color output: always, auto, never
    #[arg(long = "color", default_value = "auto", global = true)]
    pub color: String,
}

/// Available subcommands.
#[derive(Subcommand)]
pub enum Commands {
    /// Compress a GGUF file in place or to a new file
    ///
    /// Examples:
    ///   turboquant compress model.gguf -o model-turbo3.gguf
    ///   turboquant compress ~/.ollama/models/blobs/sha256-* --in-place --bits 3
    ///   turboquant compress *.gguf --block-size 128
    Compress {
        /// Input GGUF file(s) to compress
        #[arg(value_hint = ValueHint::FilePath)]
        files: Vec<String>,

        /// Output file (single file mode only)
        #[arg(short = 'o', long = "output", value_hint = ValueHint::FilePath)]
        output: Option<String>,

        /// Compress in place (overwrite input file)
        #[arg(long = "in-place")]
        in_place: bool,

        /// Number of quantization bits (default: 3)
        #[arg(long = "bits", default_value = "3")]
        bits: u8,

        /// Block size for per-block scaling (default: 64)
        #[arg(long = "block-size", default_value = "64")]
        block_size: usize,

        /// Scale mode: absmax, percentile, adaptive
        #[arg(long = "scale-mode", default_value = "absmax")]
        scale_mode: String,
    },

    /// Verify a compressed GGUF: decompress every tensor and report
    /// per-tensor SNR/MSE when the original file is provided
    Verify {
        /// GGUF file to verify
        #[arg(value_hint = ValueHint::FilePath)]
        file: String,

        /// Original (uncompressed) GGUF to compare against for SNR/MSE
        #[arg(long = "original", value_hint = ValueHint::FilePath)]
        original: Option<String>,
    },

    /// Benchmark compression/decompression vs FP16
    Bench {
        /// Head dimension (default: 128)
        #[arg(long = "head-dim", default_value = "128")]
        head_dim: usize,

        /// Sequence length (default: 4096)
        #[arg(long = "seq-len", default_value = "4096")]
        seq_len: usize,

        /// Number of attention heads (default: 32)
        #[arg(long = "num-heads", default_value = "32")]
        num_heads: usize,

        /// Number of iterations (default: 100)
        #[arg(short = 'n', default_value = "100")]
        iterations: usize,
    },

    /// Calibrate QJL scale on a representative dataset
    Calibrate {
        /// Path to calibration data (NPY format)
        #[arg(value_hint = ValueHint::FilePath)]
        data_path: String,

        /// Output YAML file for calibration parameters
        #[arg(short = 'o', long = "output", default_value = "calibration.yaml")]
        output: String,
    },

    /// Audit a folder of Ollama models and estimate gains
    Audit {
        /// Directory to audit (default: ~/.ollama/models)
        #[arg(value_hint = ValueHint::DirPath)]
        path: Option<String>,
    },

    /// Display system info: backends available, GPU detected
    Info,

    /// Start the daemon service (usually run by systemd)
    Daemon {
        /// Config file path
        #[arg(short = 'c', long = "config")]
        config: Option<String>,
    },
}

/// Custom style for the CLI help.
const fn clap_style() -> clap::builder::Styles {
    clap::builder::Styles::styled()
        .header(
            anstyle::Style::new()
                .fg_color(Some(anstyle::Color::Ansi(anstyle::AnsiColor::BrightCyan)))
                .bold(),
        )
        .usage(
            anstyle::Style::new()
                .fg_color(Some(anstyle::Color::Ansi(anstyle::AnsiColor::BrightWhite)))
                .bold(),
        )
        .literal(
            anstyle::Style::new()
                .fg_color(Some(anstyle::Color::Ansi(anstyle::AnsiColor::BrightGreen))),
        )
        .placeholder(
            anstyle::Style::new()
                .fg_color(Some(anstyle::Color::Ansi(anstyle::AnsiColor::BrightYellow))),
        )
        .error(
            anstyle::Style::new()
                .fg_color(Some(anstyle::Color::Ansi(anstyle::AnsiColor::BrightRed)))
                .bold(),
        )
        .valid(
            anstyle::Style::new()
                .fg_color(Some(anstyle::Color::Ansi(anstyle::AnsiColor::BrightGreen))),
        )
        .invalid(
            anstyle::Style::new()
                .fg_color(Some(anstyle::Color::Ansi(anstyle::AnsiColor::BrightRed))),
        )
}

fn main() {
    let cli = Cli::parse();

    // Configure tracing
    let level = match cli.verbose {
        0 => "warn",
        1 => "info",
        2 => "debug",
        _ => "trace",
    };
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(level)),
        )
        .init();

    let result = match cli.backend.as_str() {
        "cpu" | "auto" => run_command(cli.command),
        "cuda" => Err(
            "CUDA backend not implemented (this build has no GPU support; use --backend cpu)"
                .into(),
        ),
        other => Err(format!("unknown backend '{other}' (expected: cpu, auto)").into()),
    };

    if let Err(e) = result {
        eprintln!("{} {}", "error:".red().bold(), e);
        std::process::exit(1);
    }
}

/// Dispatch a parsed subcommand.
fn run_command(command: Commands) -> Result<(), Box<dyn std::error::Error>> {
    match command {
        Commands::Compress {
            files,
            output,
            in_place,
            bits,
            block_size,
            scale_mode,
        } => commands::compress::run(files, output, in_place, bits, block_size, scale_mode),
        Commands::Verify { file, original } => commands::verify::run(file, original),
        Commands::Bench {
            head_dim,
            seq_len,
            num_heads,
            iterations,
        } => commands::bench::run(head_dim, seq_len, num_heads, iterations),
        Commands::Calibrate { data_path, output } => commands::calibrate::run(data_path, output),
        Commands::Audit { path } => commands::audit::run(path),
        Commands::Info => commands::info::run(),
        Commands::Daemon { config } => commands::daemon::run(config),
    }
}
