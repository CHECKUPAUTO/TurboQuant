//! GGUF types and constants.

/// GGUF header structure.
#[derive(Debug, Clone)]
pub struct GgufHeader {
    /// Format version.
    pub version: u32,
    /// Number of tensors.
    pub tensor_count: u64,
}
