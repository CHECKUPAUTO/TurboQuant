//! GGUF file parser.

use crate::types::GgufHeader;
use turboquant_core::error::TurboQuantError;

/// Parse a GGUF file header from a byte slice.
///
/// # Errors
///
/// Returns an error if the magic number or format version is invalid.
pub fn parse_header(data: &[u8]) -> Result<GgufHeader, TurboQuantError> {
    if data.len() < 16 {
        return Err(TurboQuantError::InvalidGguf(
            "File too short for GGUF header".into(),
        ));
    }
    // GGUF magic: "GGUF" = 0x47 0x47 0x55 0x46
    if &data[0..4] != b"GGUF" {
        return Err(TurboQuantError::InvalidGguf(
            "Invalid GGUF magic number".into(),
        ));
    }
    let version = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
    let tensor_count = u64::from_le_bytes([
        data[8], data[9], data[10], data[11], data[12], data[13], data[14], data[15],
    ]);
    Ok(GgufHeader {
        version,
        tensor_count,
    })
}
