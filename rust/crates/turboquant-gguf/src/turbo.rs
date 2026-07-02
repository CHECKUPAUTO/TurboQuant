//! TurboQuant compression of GGUF files ("turbo3" format, version 1).
//!
//! Each F32/F16 tensor is quantized with [`QjlQuantizer`](turboquant_core::qjl::QjlQuantizer) (3-bit codes +
//! optional 1-bit residual correction, per-block f16 scales) and stored
//! back into a standard GGUF v3 container as a flat `I8` byte tensor,
//! with all reconstruction parameters carried in `turboquant.*` metadata
//! keys. Non-float tensors pass through unchanged. See `rust/docs/GGUF.md`
//! for the full on-disk format specification.

use crate::parser::{GgufFile, GgufParser};
use crate::types::{GgmlType, GgufValue, GgufValueType};
use crate::writer::GgufWriter;
use half::f16;
use std::path::Path;
use turboquant_core::error::TurboQuantError;
use turboquant_core::qjl::{CompressedBlock, CorrectionMode, QjlConfig, QjlQuantizer, ScaleMode};

/// Current `TurboQuant` on-disk format version.
pub const FORMAT_VERSION: u32 = 1;

/// Metadata key: format version (presence marks a file as compressed).
pub const KEY_FORMAT_VERSION: &str = "turboquant.format_version";
/// Metadata key: version of the tool that wrote the file.
pub const KEY_TOOL_VERSION: &str = "turboquant.version";
/// Metadata key: quantization bits.
pub const KEY_BITS: &str = "turboquant.bits";
/// Metadata key: block size (values per scale block).
pub const KEY_BLOCK_SIZE: &str = "turboquant.block_size";
/// Metadata key: scale mode used at compression time.
pub const KEY_SCALE_MODE: &str = "turboquant.scale_mode";
/// Metadata key: correction mode (`"none"` or `"one_bit_residual"`).
pub const KEY_CORRECTION: &str = "turboquant.correction";
/// Metadata key: learned scale of the 1-bit residual correction.
pub const KEY_CORRECTION_SCALE: &str = "turboquant.correction_scale";

fn key_orig_type(name: &str) -> String {
    format!("turboquant.{name}.orig_type")
}
fn key_orig_shape(name: &str) -> String {
    format!("turboquant.{name}.orig_shape")
}
fn key_scales(name: &str) -> String {
    format!("turboquant.{name}.scales")
}

fn err_c(msg: impl Into<String>) -> TurboQuantError {
    TurboQuantError::CompressionError(msg.into())
}
fn err_d(msg: impl Into<String>) -> TurboQuantError {
    TurboQuantError::DecompressionError(msg.into())
}

/// Options for compressing a GGUF file.
#[derive(Debug, Clone)]
pub struct TurboOptions {
    /// Quantization bits. Only 3 is currently implemented.
    pub bits: u8,
    /// Values per scale block (power of two >= 8).
    pub block_size: usize,
    /// Per-block scale computation mode.
    pub scale_mode: ScaleMode,
    /// Residual correction mode.
    pub correction: CorrectionMode,
}

impl Default for TurboOptions {
    fn default() -> Self {
        let d = QjlConfig::default();
        Self {
            bits: d.bits,
            block_size: d.block_size,
            scale_mode: d.scale_mode,
            correction: d.correction,
        }
    }
}

/// Statistics from a compression run.
#[derive(Debug, Clone, Default)]
pub struct CompressStats {
    /// Number of tensors that were quantized.
    pub tensors_compressed: usize,
    /// Number of tensors copied through unchanged.
    pub tensors_passthrough: usize,
    /// Total tensor-data bytes in the input.
    pub input_data_bytes: u64,
    /// Total tensor-data bytes in the output.
    pub output_data_bytes: u64,
}

impl CompressStats {
    /// Input/output ratio over tensor data (1.0 when output is empty).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn data_ratio(&self) -> f64 {
        if self.output_data_bytes == 0 {
            1.0
        } else {
            self.input_data_bytes as f64 / self.output_data_bytes as f64
        }
    }
}

/// Whether a parsed GGUF file is TurboQuant-compressed.
#[must_use]
pub fn is_turbo_compressed(file: &GgufFile) -> bool {
    file.metadata_value(KEY_FORMAT_VERSION).is_some()
}

/// Names of tensors in `file` that are stored TurboQuant-compressed.
#[must_use]
pub fn turbo_tensor_names(file: &GgufFile) -> Vec<String> {
    file.tensors
        .iter()
        .filter(|t| file.metadata_value(&key_orig_type(&t.name)).is_some())
        .map(|t| t.name.clone())
        .collect()
}

fn scale_mode_name(mode: &ScaleMode) -> String {
    match mode {
        ScaleMode::PerBlockAbsMax => "absmax".to_string(),
        ScaleMode::PerBlockPercentile(p) => format!("percentile:{p}"),
        ScaleMode::Adaptive => "adaptive".to_string(),
        ScaleMode::Fixed(s) => format!("fixed:{s}"),
    }
}

/// Compress all F32/F16 tensors of a parsed GGUF file, producing a
/// writer for the compressed output file.
///
/// # Errors
///
/// Returns an error if the file is already compressed, options are
/// invalid (only 3-bit is implemented), or tensor data is unreadable.
pub fn compress(
    file: &GgufFile,
    opts: &TurboOptions,
) -> Result<(GgufWriter, CompressStats), TurboQuantError> {
    if opts.bits != 3 {
        return Err(TurboQuantError::InvalidBitWidth(opts.bits));
    }
    if !opts.block_size.is_power_of_two() || opts.block_size < 8 {
        return Err(TurboQuantError::InvalidBlockSize(opts.block_size));
    }
    if is_turbo_compressed(file) {
        return Err(err_c(
            "input is already TurboQuant-compressed (turboquant.format_version present)",
        ));
    }

    let quantizer = QjlQuantizer::new(QjlConfig {
        bits: opts.bits,
        block_size: opts.block_size,
        scale_mode: opts.scale_mode.clone(),
        correction: opts.correction.clone(),
    });

    let mut writer = GgufWriter::new();
    let mut stats = CompressStats::default();

    // Carry the original metadata through.
    for (key, value) in &file.metadata {
        writer.add_metadata(key.clone(), value.clone());
    }

    // Global TurboQuant keys.
    writer.add_metadata(KEY_FORMAT_VERSION, GgufValue::U32(FORMAT_VERSION));
    writer.add_metadata(
        KEY_TOOL_VERSION,
        GgufValue::String(env!("CARGO_PKG_VERSION").to_string()),
    );
    writer.add_metadata(KEY_BITS, GgufValue::U8(opts.bits));
    writer.add_metadata(
        KEY_BLOCK_SIZE,
        GgufValue::U32(u32::try_from(opts.block_size).map_err(|_| err_c("block size overflow"))?),
    );
    writer.add_metadata(
        KEY_SCALE_MODE,
        GgufValue::String(scale_mode_name(&opts.scale_mode)),
    );
    match opts.correction {
        CorrectionMode::None => {
            writer.add_metadata(KEY_CORRECTION, GgufValue::String("none".to_string()));
        }
        CorrectionMode::OneBitResidual { learned_scale } => {
            writer.add_metadata(
                KEY_CORRECTION,
                GgufValue::String("one_bit_residual".to_string()),
            );
            writer.add_metadata(KEY_CORRECTION_SCALE, GgufValue::F32(learned_scale));
        }
    }

    for info in &file.tensors {
        let raw = file.tensor_data(info)?;
        stats.input_data_bytes += raw.len() as u64;
        match info.ggml_type {
            GgmlType::F32 | GgmlType::F16 => {
                let values = file.tensor_f32(info)?;
                let (payload, scales) = quantize_tensor(&quantizer, &values, opts.block_size);
                stats.output_data_bytes += payload.len() as u64;

                let type_name = if info.ggml_type == GgmlType::F32 {
                    "F32"
                } else {
                    "F16"
                };
                writer.add_metadata(
                    key_orig_type(&info.name),
                    GgufValue::String(type_name.to_string()),
                );
                writer.add_metadata(
                    key_orig_shape(&info.name),
                    GgufValue::Array(
                        GgufValueType::U64,
                        info.dims.iter().map(|&d| GgufValue::U64(d)).collect(),
                    ),
                );
                writer.add_metadata(
                    key_scales(&info.name),
                    GgufValue::Array(
                        GgufValueType::F32,
                        scales.into_iter().map(GgufValue::F32).collect(),
                    ),
                );
                writer.add_tensor(
                    info.name.clone(),
                    vec![payload.len() as u64],
                    GgmlType::I8,
                    payload,
                )?;
                stats.tensors_compressed += 1;
            }
            _ => {
                stats.output_data_bytes += raw.len() as u64;
                writer.add_tensor(
                    info.name.clone(),
                    info.dims.clone(),
                    info.ggml_type,
                    raw.to_vec(),
                )?;
                stats.tensors_passthrough += 1;
            }
        }
    }

    Ok((writer, stats))
}

/// Compress `input` (a GGUF file on disk) and write the result to
/// `output`.
///
/// # Errors
///
/// Returns an error on I/O failure, malformed input, or if the input is
/// already compressed.
pub fn compress_file(
    input: impl AsRef<Path>,
    output: impl AsRef<Path>,
    opts: &TurboOptions,
) -> Result<CompressStats, TurboQuantError> {
    let file = GgufParser::parse_file(input)?;
    let (writer, stats) = compress(&file, opts)?;
    writer.write_to_file(output)?;
    Ok(stats)
}

/// Quantize one tensor's values block-by-block.
///
/// Payload layout: all packed 3-bit code sections (block-by-block),
/// followed by all correction-bit sections (block-by-block, only when
/// correction is enabled). Returns `(payload, per-block scales)`.
fn quantize_tensor(
    quantizer: &QjlQuantizer,
    values: &[f32],
    block_size: usize,
) -> (Vec<u8>, Vec<f32>) {
    let mut packed = Vec::with_capacity(values.len() * 3 / 8 + 8);
    let mut corrections = Vec::new();
    let mut scales = Vec::with_capacity(values.len().div_ceil(block_size.max(1)));
    for chunk in values.chunks(block_size) {
        let block = quantizer.quantize_block(chunk);
        scales.push(block.scale.to_f32());
        packed.extend_from_slice(&block.packed);
        if let Some(bits) = block.correction_bits {
            corrections.extend_from_slice(&bits);
        }
    }
    packed.extend_from_slice(&corrections);
    (packed, scales)
}

/// Quantization parameters read back from a compressed file's metadata.
#[derive(Debug, Clone)]
pub struct TurboParams {
    /// Quantization bits.
    pub bits: u8,
    /// Values per scale block.
    pub block_size: usize,
    /// Correction mode.
    pub correction: CorrectionMode,
}

/// Read the global `TurboQuant` parameters from a compressed file.
///
/// # Errors
///
/// Returns an error if the file is not TurboQuant-compressed or its
/// metadata is inconsistent.
pub fn read_params(file: &GgufFile) -> Result<TurboParams, TurboQuantError> {
    let version = file
        .metadata_value(KEY_FORMAT_VERSION)
        .and_then(GgufValue::as_u64)
        .ok_or_else(|| err_d("not a TurboQuant-compressed file (missing format version)"))?;
    if version != u64::from(FORMAT_VERSION) {
        return Err(err_d(format!(
            "unsupported TurboQuant format version {version} (supported: {FORMAT_VERSION})"
        )));
    }
    let bits = file
        .metadata_value(KEY_BITS)
        .and_then(GgufValue::as_u64)
        .ok_or_else(|| err_d("missing turboquant.bits"))?;
    if bits != 3 {
        return Err(err_d(format!("unsupported bit width {bits} (only 3)")));
    }
    let block_size = file
        .metadata_value(KEY_BLOCK_SIZE)
        .and_then(GgufValue::as_u64)
        .ok_or_else(|| err_d("missing turboquant.block_size"))?;
    let block_size = usize::try_from(block_size).map_err(|_| err_d("block size overflow"))?;
    if !block_size.is_power_of_two() || block_size < 8 {
        return Err(TurboQuantError::InvalidBlockSize(block_size));
    }
    let correction = match file
        .metadata_value(KEY_CORRECTION)
        .and_then(GgufValue::as_str)
    {
        Some("none") | None => CorrectionMode::None,
        Some("one_bit_residual") => {
            let learned_scale = file
                .metadata_value(KEY_CORRECTION_SCALE)
                .and_then(GgufValue::as_f32)
                .ok_or_else(|| err_d("missing turboquant.correction_scale"))?;
            CorrectionMode::OneBitResidual { learned_scale }
        }
        Some(other) => return Err(err_d(format!("unknown correction mode '{other}'"))),
    };
    Ok(TurboParams {
        bits: 3,
        block_size,
        correction,
    })
}

/// Original type and shape of a compressed tensor.
#[derive(Debug, Clone)]
pub struct TurboTensorMeta {
    /// Original element type (`F32` or `F16`).
    pub orig_type: GgmlType,
    /// Original dimensions (ggml order).
    pub orig_shape: Vec<u64>,
    /// Per-block scales.
    pub scales: Vec<f32>,
}

/// Read the per-tensor `TurboQuant` metadata for tensor `name`.
///
/// # Errors
///
/// Returns an error if the tensor is not stored compressed or its
/// metadata is malformed.
pub fn read_tensor_meta(file: &GgufFile, name: &str) -> Result<TurboTensorMeta, TurboQuantError> {
    let orig_type = match file
        .metadata_value(&key_orig_type(name))
        .and_then(GgufValue::as_str)
    {
        Some("F32") => GgmlType::F32,
        Some("F16") => GgmlType::F16,
        Some(other) => return Err(err_d(format!("unknown orig_type '{other}' for '{name}'"))),
        None => {
            return Err(err_d(format!(
                "tensor '{name}' is not TurboQuant-compressed (missing orig_type)"
            )))
        }
    };
    let orig_shape: Vec<u64> = file
        .metadata_value(&key_orig_shape(name))
        .and_then(GgufValue::as_array)
        .ok_or_else(|| err_d(format!("missing orig_shape for '{name}'")))?
        .iter()
        .map(|v| {
            v.as_u64()
                .ok_or_else(|| err_d("orig_shape must be integers"))
        })
        .collect::<Result<_, _>>()?;
    let scales: Vec<f32> = file
        .metadata_value(&key_scales(name))
        .and_then(GgufValue::as_array)
        .ok_or_else(|| err_d(format!("missing scales for '{name}'")))?
        .iter()
        .map(|v| v.as_f32().ok_or_else(|| err_d("scales must be f32")))
        .collect::<Result<_, _>>()?;
    Ok(TurboTensorMeta {
        orig_type,
        orig_shape,
        scales,
    })
}

/// Decompress a TurboQuant-compressed tensor back to `f32` values.
///
/// # Errors
///
/// Returns an error if the tensor or its metadata is missing/malformed.
pub fn decompress_tensor(file: &GgufFile, name: &str) -> Result<Vec<f32>, TurboQuantError> {
    let params = read_params(file)?;
    let meta = read_tensor_meta(file, name)?;
    let info = file
        .tensor(name)
        .ok_or_else(|| err_d(format!("tensor '{name}' not found")))?;
    if info.ggml_type != GgmlType::I8 {
        return Err(err_d(format!(
            "compressed tensor '{name}' must be stored as I8 bytes, found {:?}",
            info.ggml_type
        )));
    }
    let payload = file.tensor_data(info)?;

    let n = usize::try_from(meta.orig_shape.iter().product::<u64>())
        .map_err(|_| err_d("element count overflow"))?;
    let block_size = params.block_size;
    let n_blocks = n.div_ceil(block_size);
    if meta.scales.len() != n_blocks {
        return Err(err_d(format!(
            "tensor '{name}': expected {n_blocks} scales, found {}",
            meta.scales.len()
        )));
    }
    let has_correction = matches!(params.correction, CorrectionMode::OneBitResidual { .. });

    // Compute section sizes.
    let block_len = |i: usize| -> usize {
        if i + 1 == n_blocks && n % block_size != 0 {
            n % block_size
        } else {
            block_size
        }
    };
    let packed_len = |values: usize| values.next_multiple_of(8) * 3 / 8;
    let corr_len = |values: usize| values.div_ceil(8);
    let total_packed: usize = (0..n_blocks).map(|i| packed_len(block_len(i))).sum();
    let total_corr: usize = if has_correction {
        (0..n_blocks).map(|i| corr_len(block_len(i))).sum()
    } else {
        0
    };
    if payload.len() != total_packed + total_corr {
        return Err(err_d(format!(
            "tensor '{name}': payload is {} bytes, expected {}",
            payload.len(),
            total_packed + total_corr
        )));
    }

    let quantizer = QjlQuantizer::new(QjlConfig {
        bits: params.bits,
        block_size,
        scale_mode: ScaleMode::PerBlockAbsMax, // irrelevant for dequantization
        correction: params.correction.clone(),
    });

    let mut out = Vec::with_capacity(n);
    let mut packed_pos = 0usize;
    let mut corr_pos = total_packed;
    for (i, &scale) in meta.scales.iter().enumerate() {
        let n_in = block_len(i);
        let p_len = packed_len(n_in);
        let block = CompressedBlock {
            packed: payload[packed_pos..packed_pos + p_len].to_vec(),
            scale: f16::from_f32(scale),
            correction_bits: if has_correction {
                let c_len = corr_len(n_in);
                let bits = payload[corr_pos..corr_pos + c_len].to_vec();
                corr_pos += c_len;
                Some(bits)
            } else {
                None
            },
        };
        packed_pos += p_len;
        out.extend(quantizer.dequantize_block(&block, n_in));
    }
    Ok(out)
}

/// Mean squared error between two equal-length slices.
///
/// # Panics
///
/// Panics if the slices have different lengths.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn mse(reference: &[f32], approx: &[f32]) -> f64 {
    assert_eq!(reference.len(), approx.len(), "length mismatch");
    if reference.is_empty() {
        return 0.0;
    }
    reference
        .iter()
        .zip(approx)
        .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
        .sum::<f64>()
        / reference.len() as f64
}

/// Signal-to-noise ratio in dB of `approx` relative to `reference`.
/// Returns `f64::INFINITY` for an exact match.
///
/// # Panics
///
/// Panics if the slices have different lengths.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn snr_db(reference: &[f32], approx: &[f32]) -> f64 {
    let noise = mse(reference, approx);
    if noise == 0.0 {
        return f64::INFINITY;
    }
    let signal = reference.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>()
        / reference.len().max(1) as f64;
    10.0 * (signal / noise).log10()
}
