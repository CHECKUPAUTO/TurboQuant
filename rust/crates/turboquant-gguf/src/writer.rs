//! GGUF v3 file writer.

use crate::types::{GgmlType, GgufValue, DEFAULT_ALIGNMENT, GGUF_MAGIC, GGUF_VERSION};
use std::io::Write;
use std::path::Path;
use turboquant_core::error::TurboQuantError;

fn err(msg: impl Into<String>) -> TurboQuantError {
    TurboQuantError::InvalidGguf(msg.into())
}

struct TensorEntry {
    name: String,
    dims: Vec<u64>,
    ggml_type: GgmlType,
    data: Vec<u8>,
}

/// Builder that assembles a GGUF v3 file in memory.
///
/// Add metadata and tensors, then serialize with [`GgufWriter::to_bytes`]
/// or [`GgufWriter::write_to_file`]. Tensor data is written in insertion
/// order, each tensor aligned to the configured alignment (default 32).
pub struct GgufWriter {
    metadata: Vec<(String, GgufValue)>,
    tensors: Vec<TensorEntry>,
    alignment: u64,
}

impl GgufWriter {
    /// Create a new, empty GGUF writer.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            metadata: Vec::new(),
            tensors: Vec::new(),
            alignment: DEFAULT_ALIGNMENT,
        }
    }

    /// Set a custom data alignment. A `general.alignment` metadata key is
    /// emitted automatically when it differs from the default.
    ///
    /// # Errors
    ///
    /// Returns an error unless `alignment` is a power of two >= 8.
    pub fn set_alignment(&mut self, alignment: u64) -> Result<(), TurboQuantError> {
        if !alignment.is_power_of_two() || alignment < 8 {
            return Err(err(format!(
                "alignment must be a power of two >= 8, got {alignment}"
            )));
        }
        self.alignment = alignment;
        Ok(())
    }

    /// Add (or replace) a metadata key/value pair.
    pub fn add_metadata(&mut self, key: impl Into<String>, value: GgufValue) {
        let key = key.into();
        if let Some(slot) = self.metadata.iter_mut().find(|(k, _)| *k == key) {
            slot.1 = value;
        } else {
            self.metadata.push((key, value));
        }
    }

    /// Add a tensor. `dims` are in ggml order (innermost first). For
    /// element types with a known size, `data.len()` must match the
    /// product of dims exactly.
    ///
    /// # Errors
    ///
    /// Returns an error on duplicate names or size mismatch.
    pub fn add_tensor(
        &mut self,
        name: impl Into<String>,
        dims: Vec<u64>,
        ggml_type: GgmlType,
        data: Vec<u8>,
    ) -> Result<(), TurboQuantError> {
        let name = name.into();
        if self.tensors.iter().any(|t| t.name == name) {
            return Err(err(format!("duplicate tensor name '{name}'")));
        }
        if dims.is_empty() || dims.len() > 8 {
            return Err(err(format!(
                "tensor '{name}' must have 1-8 dims, got {}",
                dims.len()
            )));
        }
        if let Some(size) = ggml_type.element_size() {
            let expected = dims.iter().product::<u64>() * size as u64;
            if expected != data.len() as u64 {
                return Err(err(format!(
                    "tensor '{name}': expected {expected} data bytes for dims {dims:?}, got {}",
                    data.len()
                )));
            }
        }
        self.tensors.push(TensorEntry {
            name,
            dims,
            ggml_type,
            data,
        });
        Ok(())
    }

    /// Number of tensors added so far.
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    /// Serialize the file to bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if serialization fails (e.g. oversized strings).
    pub fn to_bytes(&self) -> Result<Vec<u8>, TurboQuantError> {
        let mut metadata: Vec<(&str, &GgufValue)> = Vec::new();
        let alignment_kv;
        let has_alignment_key = self.metadata.iter().any(|(k, _)| k == "general.alignment");
        if self.alignment != DEFAULT_ALIGNMENT && !has_alignment_key {
            alignment_kv = GgufValue::U32(
                u32::try_from(self.alignment).map_err(|_| err("alignment too large for u32"))?,
            );
            metadata.push(("general.alignment", &alignment_kv));
        }
        metadata.extend(self.metadata.iter().map(|(k, v)| (k.as_str(), v)));

        let mut out = Vec::new();
        out.extend_from_slice(&GGUF_MAGIC);
        out.extend_from_slice(&GGUF_VERSION.to_le_bytes());
        out.extend_from_slice(&(self.tensors.len() as u64).to_le_bytes());
        out.extend_from_slice(&(metadata.len() as u64).to_le_bytes());

        for (key, value) in &metadata {
            write_string(&mut out, key);
            out.extend_from_slice(&value.value_type().to_u32().to_le_bytes());
            write_value(&mut out, value)?;
        }

        // Assign tensor offsets (relative to data section start).
        let mut offsets = Vec::with_capacity(self.tensors.len());
        let mut cursor = 0u64;
        for t in &self.tensors {
            cursor = cursor.next_multiple_of(self.alignment);
            offsets.push(cursor);
            cursor += t.data.len() as u64;
        }

        for (t, offset) in self.tensors.iter().zip(&offsets) {
            write_string(&mut out, &t.name);
            out.extend_from_slice(&(t.dims.len() as u32).to_le_bytes());
            for d in &t.dims {
                out.extend_from_slice(&d.to_le_bytes());
            }
            out.extend_from_slice(&t.ggml_type.to_u32().to_le_bytes());
            out.extend_from_slice(&offset.to_le_bytes());
        }

        // Pad to the aligned data section start, then write tensor data.
        let data_start = (out.len() as u64).next_multiple_of(self.alignment);
        out.resize(
            usize::try_from(data_start).map_err(|_| err("file too large"))?,
            0,
        );
        for (t, offset) in self.tensors.iter().zip(&offsets) {
            let abs = usize::try_from(data_start + offset).map_err(|_| err("file too large"))?;
            out.resize(abs, 0); // inter-tensor alignment padding
            out.extend_from_slice(&t.data);
        }
        Ok(out)
    }

    /// Serialize and write the file to disk.
    ///
    /// # Errors
    ///
    /// Returns an error on serialization or I/O failure.
    pub fn write_to_file(&self, path: impl AsRef<Path>) -> Result<(), TurboQuantError> {
        let bytes = self.to_bytes()?;
        let mut f = std::fs::File::create(path)?;
        f.write_all(&bytes)?;
        f.flush()?;
        Ok(())
    }
}

impl Default for GgufWriter {
    fn default() -> Self {
        Self::new()
    }
}

fn write_string(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u64).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

#[allow(clippy::cast_sign_loss)]
fn write_value(out: &mut Vec<u8>, value: &GgufValue) -> Result<(), TurboQuantError> {
    match value {
        GgufValue::U8(v) => out.push(*v),
        GgufValue::I8(v) => out.push(*v as u8),
        GgufValue::U16(v) => out.extend_from_slice(&v.to_le_bytes()),
        GgufValue::I16(v) => out.extend_from_slice(&v.to_le_bytes()),
        GgufValue::U32(v) => out.extend_from_slice(&v.to_le_bytes()),
        GgufValue::I32(v) => out.extend_from_slice(&v.to_le_bytes()),
        GgufValue::F32(v) => out.extend_from_slice(&v.to_le_bytes()),
        GgufValue::Bool(v) => out.push(u8::from(*v)),
        GgufValue::String(s) => write_string(out, s),
        GgufValue::U64(v) => out.extend_from_slice(&v.to_le_bytes()),
        GgufValue::I64(v) => out.extend_from_slice(&v.to_le_bytes()),
        GgufValue::F64(v) => out.extend_from_slice(&v.to_le_bytes()),
        GgufValue::Array(elem_ty, values) => {
            for v in values {
                if v.value_type() != *elem_ty {
                    return Err(err(format!(
                        "array element type {:?} does not match declared {elem_ty:?}",
                        v.value_type()
                    )));
                }
            }
            out.extend_from_slice(&elem_ty.to_u32().to_le_bytes());
            out.extend_from_slice(&(values.len() as u64).to_le_bytes());
            for v in values {
                write_value(out, v)?;
            }
        }
    }
    Ok(())
}
