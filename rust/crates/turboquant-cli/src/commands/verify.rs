//! Verify command implementation.
//!
//! Parses a GGUF file, and if it is TurboQuant-compressed, decompresses
//! every compressed tensor. When `--original` is given, reports real
//! per-tensor SNR/MSE against the uncompressed source model.

use colored::Colorize;
use turboquant_gguf::turbo;
use turboquant_gguf::{GgufFile, GgufParser};

/// Run the verify command.
///
/// # Errors
///
/// Returns an error if the file is malformed or any tensor fails to
/// decompress.
pub fn run(file: String, original: Option<String>) -> Result<(), Box<dyn std::error::Error>> {
    let path = std::path::Path::new(&file);
    if !path.exists() {
        return Err(format!("File not found: {file}").into());
    }
    let file_size = std::fs::metadata(path)?.len();
    let parsed = GgufParser::parse_file(path)?;

    println!("{} GGUF container parsed", "OK".green().bold());
    println!("  File:      {}", file.cyan());
    println!("  Size:      {file_size} bytes");
    println!("  Version:   {}", parsed.header.version.to_string().cyan());
    println!("  Tensors:   {}", parsed.header.tensor_count);
    println!("  Metadata:  {} keys", parsed.header.metadata_kv_count);
    println!("  Alignment: {}", parsed.alignment);

    if !turbo::is_turbo_compressed(&parsed) {
        println!();
        println!(
            "{} Not a TurboQuant-compressed file (container structure is valid)",
            "note:".yellow().bold()
        );
        return Ok(());
    }

    let params = turbo::read_params(&parsed)?;
    println!();
    println!("{}", "TurboQuant format".bold().cyan());
    println!("  Bits:       {}", params.bits);
    println!("  Block size: {}", params.block_size);

    let orig = match &original {
        Some(p) => Some(GgufParser::parse_file(p)?),
        None => None,
    };

    let names = turbo::turbo_tensor_names(&parsed);
    println!("  Compressed tensors: {}", names.len());
    println!();

    let mut min_snr = f64::INFINITY;
    for name in &names {
        let recon = turbo::decompress_tensor(&parsed, name)
            .map_err(|e| format!("tensor '{name}' failed to decompress: {e}"))?;
        if let Some(orig_file) = &orig {
            let (mse, snr) = compare_with_original(orig_file, name, &recon)?;
            min_snr = min_snr.min(snr);
            println!(
                "  {} {:<40} {:>10} values  MSE {:>12.6e}  SNR {:>7.2} dB",
                "OK".green().bold(),
                name,
                recon.len(),
                mse,
                snr
            );
        } else {
            let rms = (recon.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>()
                / recon.len().max(1) as f64)
                .sqrt();
            println!(
                "  {} {:<40} {:>10} values decoded  rms {:.4}",
                "OK".green().bold(),
                name,
                recon.len(),
                rms
            );
        }
    }

    println!();
    if orig.is_some() {
        if names.is_empty() {
            println!("{} No compressed tensors to check", "note:".yellow().bold());
        } else if min_snr > 10.0 {
            println!(
                "{} All {} tensor(s) decompressed; minimum SNR {:.2} dB",
                "OK".green().bold(),
                names.len(),
                min_snr
            );
        } else {
            return Err(format!(
                "minimum SNR {min_snr:.2} dB is below the 10 dB plausibility threshold"
            )
            .into());
        }
    } else {
        println!(
            "{} All {} tensor(s) decompressed successfully",
            "OK".green().bold(),
            names.len()
        );
        println!(
            "{} pass --original <file> to compute SNR/MSE against the source model",
            "note:".yellow().bold()
        );
    }
    Ok(())
}

/// Compute (MSE, SNR dB) of `recon` against tensor `name` in the
/// original file.
fn compare_with_original(
    orig: &GgufFile,
    name: &str,
    recon: &[f32],
) -> Result<(f64, f64), Box<dyn std::error::Error>> {
    let info = orig
        .tensor(name)
        .ok_or_else(|| format!("tensor '{name}' not present in --original file"))?;
    let reference = orig
        .tensor_f32(info)
        .map_err(|e| format!("cannot decode original tensor '{name}': {e}"))?;
    if reference.len() != recon.len() {
        return Err(format!(
            "tensor '{name}': element count mismatch (original {}, reconstructed {})",
            reference.len(),
            recon.len()
        )
        .into());
    }
    Ok((
        turbo::mse(&reference, recon),
        turbo::snr_db(&reference, recon),
    ))
}
