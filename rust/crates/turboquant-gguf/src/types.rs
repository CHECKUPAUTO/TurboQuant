//! GGUF types and constants.

use turboquant_core::error::TurboQuantError;

/// GGUF magic bytes ("GGUF").
pub const GGUF_MAGIC: [u8; 4] = *b"GGUF";

/// GGUF version written by this crate.
pub const GGUF_VERSION: u32 = 3;

/// Default alignment (in bytes) of the tensor-data section and of each
/// tensor within it, per the GGUF spec. Overridable via the
/// `general.alignment` metadata key.
pub const DEFAULT_ALIGNMENT: u64 = 32;

/// GGUF header structure (magic excluded).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgufHeader {
    /// Format version.
    pub version: u32,
    /// Number of tensors.
    pub tensor_count: u64,
    /// Number of metadata key/value pairs.
    pub metadata_kv_count: u64,
}

/// Wire type ids for GGUF metadata values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GgufValueType {
    /// 8-bit unsigned integer.
    U8 = 0,
    /// 8-bit signed integer.
    I8 = 1,
    /// 16-bit unsigned integer.
    U16 = 2,
    /// 16-bit signed integer.
    I16 = 3,
    /// 32-bit unsigned integer.
    U32 = 4,
    /// 32-bit signed integer.
    I32 = 5,
    /// 32-bit IEEE 754 float.
    F32 = 6,
    /// Boolean (1 byte, 0 or 1).
    Bool = 7,
    /// UTF-8 string (u64 length prefix, no terminator).
    String = 8,
    /// Homogeneous array (elem type u32 + count u64 + elements).
    Array = 9,
    /// 64-bit unsigned integer.
    U64 = 10,
    /// 64-bit signed integer.
    I64 = 11,
    /// 64-bit IEEE 754 float.
    F64 = 12,
}

impl GgufValueType {
    /// Decode a wire type id.
    ///
    /// # Errors
    ///
    /// Returns an error for ids outside `0..=12`.
    pub fn from_u32(v: u32) -> Result<Self, TurboQuantError> {
        Ok(match v {
            0 => Self::U8,
            1 => Self::I8,
            2 => Self::U16,
            3 => Self::I16,
            4 => Self::U32,
            5 => Self::I32,
            6 => Self::F32,
            7 => Self::Bool,
            8 => Self::String,
            9 => Self::Array,
            10 => Self::U64,
            11 => Self::I64,
            12 => Self::F64,
            other => {
                return Err(TurboQuantError::InvalidGguf(format!(
                    "unknown GGUF metadata value type id {other}"
                )))
            }
        })
    }

    /// Encode to the wire type id.
    #[must_use]
    pub const fn to_u32(self) -> u32 {
        self as u32
    }
}

/// A GGUF metadata value.
#[derive(Debug, Clone, PartialEq)]
pub enum GgufValue {
    /// 8-bit unsigned integer.
    U8(u8),
    /// 8-bit signed integer.
    I8(i8),
    /// 16-bit unsigned integer.
    U16(u16),
    /// 16-bit signed integer.
    I16(i16),
    /// 32-bit unsigned integer.
    U32(u32),
    /// 32-bit signed integer.
    I32(i32),
    /// 32-bit float.
    F32(f32),
    /// Boolean.
    Bool(bool),
    /// UTF-8 string.
    String(String),
    /// Homogeneous array: element type + elements.
    Array(GgufValueType, Vec<GgufValue>),
    /// 64-bit unsigned integer.
    U64(u64),
    /// 64-bit signed integer.
    I64(i64),
    /// 64-bit float.
    F64(f64),
}

impl GgufValue {
    /// The wire type of this value.
    #[must_use]
    pub const fn value_type(&self) -> GgufValueType {
        match self {
            Self::U8(_) => GgufValueType::U8,
            Self::I8(_) => GgufValueType::I8,
            Self::U16(_) => GgufValueType::U16,
            Self::I16(_) => GgufValueType::I16,
            Self::U32(_) => GgufValueType::U32,
            Self::I32(_) => GgufValueType::I32,
            Self::F32(_) => GgufValueType::F32,
            Self::Bool(_) => GgufValueType::Bool,
            Self::String(_) => GgufValueType::String,
            Self::Array(_, _) => GgufValueType::Array,
            Self::U64(_) => GgufValueType::U64,
            Self::I64(_) => GgufValueType::I64,
            Self::F64(_) => GgufValueType::F64,
        }
    }

    /// Interpret any integer variant as `u64` (signed variants must be
    /// non-negative). Returns `None` for non-integer variants.
    #[must_use]
    #[allow(clippy::cast_sign_loss)]
    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            Self::U8(v) => Some(u64::from(v)),
            Self::U16(v) => Some(u64::from(v)),
            Self::U32(v) => Some(u64::from(v)),
            Self::U64(v) => Some(v),
            Self::I8(v) if v >= 0 => Some(v as u64),
            Self::I16(v) if v >= 0 => Some(v as u64),
            Self::I32(v) if v >= 0 => Some(v as u64),
            Self::I64(v) if v >= 0 => Some(v as u64),
            _ => None,
        }
    }

    /// Get the value as an `f32` if it is a float variant.
    #[must_use]
    pub const fn as_f32(&self) -> Option<f32> {
        match *self {
            Self::F32(v) => Some(v),
            _ => None,
        }
    }

    /// Get the value as a string slice if it is a string.
    #[must_use]
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Get the array elements if this is an array.
    #[must_use]
    pub fn as_array(&self) -> Option<&[GgufValue]> {
        match self {
            Self::Array(_, v) => Some(v.as_slice()),
            _ => None,
        }
    }
}

/// GGML tensor data types (subset relevant to `TurboQuant`, plus a raw
/// fallback for anything else).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(missing_docs)] // variant names mirror the ggml enum
pub enum GgmlType {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
    Iq2Xxs,
    Iq2Xs,
    Iq3Xxs,
    Iq1S,
    Iq4Nl,
    Iq3S,
    Iq2S,
    Iq4Xs,
    I8,
    I16,
    I32,
    I64,
    F64,
    Iq1M,
    Bf16,
    /// Any ggml type id this crate does not know; data is exposed as raw
    /// bytes only.
    Unknown(u32),
}

impl GgmlType {
    /// Decode a ggml type id.
    #[must_use]
    pub const fn from_u32(v: u32) -> Self {
        match v {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            9 => Self::Q8_1,
            10 => Self::Q2K,
            11 => Self::Q3K,
            12 => Self::Q4K,
            13 => Self::Q5K,
            14 => Self::Q6K,
            15 => Self::Q8K,
            16 => Self::Iq2Xxs,
            17 => Self::Iq2Xs,
            18 => Self::Iq3Xxs,
            19 => Self::Iq1S,
            20 => Self::Iq4Nl,
            21 => Self::Iq3S,
            22 => Self::Iq2S,
            23 => Self::Iq4Xs,
            24 => Self::I8,
            25 => Self::I16,
            26 => Self::I32,
            27 => Self::I64,
            28 => Self::F64,
            29 => Self::Iq1M,
            30 => Self::Bf16,
            other => Self::Unknown(other),
        }
    }

    /// Encode to the ggml type id.
    #[must_use]
    pub const fn to_u32(self) -> u32 {
        match self {
            Self::F32 => 0,
            Self::F16 => 1,
            Self::Q4_0 => 2,
            Self::Q4_1 => 3,
            Self::Q5_0 => 6,
            Self::Q5_1 => 7,
            Self::Q8_0 => 8,
            Self::Q8_1 => 9,
            Self::Q2K => 10,
            Self::Q3K => 11,
            Self::Q4K => 12,
            Self::Q5K => 13,
            Self::Q6K => 14,
            Self::Q8K => 15,
            Self::Iq2Xxs => 16,
            Self::Iq2Xs => 17,
            Self::Iq3Xxs => 18,
            Self::Iq1S => 19,
            Self::Iq4Nl => 20,
            Self::Iq3S => 21,
            Self::Iq2S => 22,
            Self::Iq4Xs => 23,
            Self::I8 => 24,
            Self::I16 => 25,
            Self::I32 => 26,
            Self::I64 => 27,
            Self::F64 => 28,
            Self::Iq1M => 29,
            Self::Bf16 => 30,
            Self::Unknown(v) => v,
        }
    }

    /// Bytes per element for simple (non-block-quantized) types, `None`
    /// for block-quantized or unknown types.
    #[must_use]
    pub const fn element_size(self) -> Option<usize> {
        match self {
            Self::I8 => Some(1),
            Self::F16 | Self::Bf16 | Self::I16 => Some(2),
            Self::F32 | Self::I32 => Some(4),
            Self::F64 | Self::I64 => Some(8),
            _ => None,
        }
    }
}

/// Metadata describing one tensor in a GGUF file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgufTensorInfo {
    /// Tensor name (max 64 bytes per spec; not enforced on read).
    pub name: String,
    /// Dimension sizes, innermost first (ggml order).
    pub dims: Vec<u64>,
    /// Element type.
    pub ggml_type: GgmlType,
    /// Byte offset of the tensor data, relative to the start of the
    /// (aligned) data section.
    pub offset: u64,
}

impl GgufTensorInfo {
    /// Total number of elements (product of dims; 1 for scalars).
    #[must_use]
    pub fn num_elements(&self) -> u64 {
        self.dims.iter().product()
    }

    /// Exact data size in bytes if the element type has a known size.
    #[must_use]
    pub fn data_size(&self) -> Option<u64> {
        self.ggml_type
            .element_size()
            .map(|s| self.num_elements() * s as u64)
    }
}
