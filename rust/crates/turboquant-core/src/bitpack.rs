//! 3-bit packing and unpacking.
//!
//! Packs 8 × 3-bit values into 3 bytes (24 bits).
//! Unpacks 3 bytes (24 bits) into 8 × 3-bit values.

/// Packs 8 × 3-bit values into 3 bytes.
///
/// # Examples
///
/// ```
/// use turboquant_core::bitpack::pack_3bit;
///
/// let values: [u8; 8] = [0, 1, 2, 3, 4, 5, 6, 7];
/// let mut out = [0u8; 3];
/// pack_3bit(&values, &mut out);
/// ```
#[inline]
#[allow(clippy::cast_possible_truncation)]
pub fn pack_3bit(values: &[u8], out: &mut [u8]) {
    assert_eq!(values.len(), 8, "pack_3bit requires exactly 8 values");
    assert_eq!(out.len(), 3, "pack_3bit output requires 3 bytes");
    // Pack: [v0 v1 v2 v3 v4 v5 v6 v7] (3 bits each) → 24 bits = 3 bytes
    // Byte 0: v0[2:0] | v1[2:0]<<3 | v2[1:0]<<6
    // Byte 1: v2[2] | v3[2:0]<<1 | v4[2:0]<<4 | v5[1:0]<<7
    // Byte 2: v5[2] | v6[2:0]<<1 | v7[2:0]<<4
    let v0 = values[0] & 0x07;
    let v1 = values[1] & 0x07;
    let v2 = values[2] & 0x07;
    let v3 = values[3] & 0x07;
    let v4 = values[4] & 0x07;
    let v5 = values[5] & 0x07;
    let v6 = values[6] & 0x07;
    let v7 = values[7] & 0x07;

    out[0] = v0 | (v1 << 3) | (v2 << 6);
    out[1] = (v2 >> 2) | (v3 << 1) | (v4 << 4) | (v5 << 7);
    out[2] = (v5 >> 1) | (v6 << 2) | (v7 << 5);
}

/// Unpacks 3 bytes (24 bits) into 8 × 3-bit values.
///
/// # Examples
///
/// ```
/// use turboquant_core::bitpack::{pack_3bit, unpack_3bit};
///
/// let values: [u8; 8] = [0, 1, 2, 3, 4, 5, 6, 7];
/// let mut packed = [0u8; 3];
/// pack_3bit(&values, &mut packed);
/// let mut unpacked = [0u8; 8];
/// unpack_3bit(&packed, &mut unpacked);
/// assert_eq!(values, unpacked);
/// ```
#[inline]
pub fn unpack_3bit(packed: &[u8], out: &mut [u8]) {
    assert_eq!(packed.len(), 3, "unpack_3bit requires exactly 3 bytes");
    assert_eq!(out.len(), 8, "unpack_3bit output requires 8 values");

    out[0] = packed[0] & 0x07;
    out[1] = (packed[0] >> 3) & 0x07;
    out[2] = ((packed[0] >> 6) & 0x03) | ((packed[1] & 0x01) << 2);
    out[3] = (packed[1] >> 1) & 0x07;
    out[4] = (packed[1] >> 4) & 0x07;
    out[5] = ((packed[1] >> 7) & 0x01) | ((packed[2] & 0x03) << 1);
    out[6] = (packed[2] >> 2) & 0x07;
    out[7] = (packed[2] >> 5) & 0x07;
}

/// Packs a slice of 3-bit values (multiple of 8) into a `Vec<u8>`.
///
/// # Errors
///
/// Returns an error if `values.len()` is not a multiple of 8.
///
/// # Examples
///
/// ```
/// use turboquant_core::bitpack::pack_3bit_slice;
///
/// let values = vec![0u8; 64];
/// let packed = pack_3bit_slice(&values).unwrap();
/// assert_eq!(packed.len(), 24); // 64 values * 3 bits / 8 = 24 bytes
/// ```
pub fn pack_3bit_slice(values: &[u8]) -> crate::Result<Vec<u8>> {
    if values.len() % 8 != 0 {
        return Err(crate::error::TurboQuantError::UnalignedData { len: values.len() });
    }
    let mut out = vec![0u8; values.len() * 3 / 8];
    for (i, chunk) in values.chunks_exact(8).enumerate() {
        pack_3bit(chunk, &mut out[i * 3..(i + 1) * 3]);
    }
    Ok(out)
}

/// Unpacks a packed byte slice into `n` 3-bit values.
///
/// # Errors
///
/// Returns an error if `n` is not a multiple of 8.
///
/// # Examples
///
/// ```
/// use turboquant_core::bitpack::{pack_3bit_slice, unpack_3bit_slice};
///
/// let values = vec![0u8; 64];
/// let packed = pack_3bit_slice(&values).unwrap();
/// let unpacked = unpack_3bit_slice(&packed, 64).unwrap();
/// assert_eq!(values, unpacked);
/// ```
pub fn unpack_3bit_slice(packed: &[u8], n: usize) -> crate::Result<Vec<u8>> {
    if n % 8 != 0 {
        return Err(crate::error::TurboQuantError::UnalignedData { len: n });
    }
    let expected_len = n * 3 / 8;
    if packed.len() < expected_len {
        return Err(crate::error::TurboQuantError::DecompressionError(
            "packed data shorter than expected".into(),
        ));
    }
    let mut out = vec![0u8; n];
    for (i, chunk) in out.chunks_exact_mut(8).enumerate() {
        unpack_3bit(&packed[i * 3..(i + 1) * 3], chunk);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_single_block() {
        let values: [u8; 8] = [0, 1, 2, 3, 4, 5, 6, 7];
        let mut packed = [0u8; 3];
        pack_3bit(&values, &mut packed);
        let mut unpacked = [0u8; 8];
        unpack_3bit(&packed, &mut unpacked);
        assert_eq!(values, unpacked);
    }

    #[test]
    fn roundtrip_all_values_0_7() {
        for a in 0..8u8 {
            for b in 0..8u8 {
                for c in 0..8u8 {
                    for d in 0..8u8 {
                        for e in 0..8u8 {
                            for f in 0..8u8 {
                                for g in 0..8u8 {
                                    for h in 0..8u8 {
                                        let vals = [a, b, c, d, e, f, g, h];
                                        let mut packed = [0u8; 3];
                                        pack_3bit(&vals, &mut packed);
                                        let mut unpacked = [0u8; 8];
                                        unpack_3bit(&packed, &mut unpacked);
                                        assert_eq!(vals, unpacked);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn roundtrip_slice_64() {
        let values: Vec<u8> = (0..64u8).map(|x| x % 8).collect();
        let packed = pack_3bit_slice(&values).unwrap();
        assert_eq!(packed.len(), 24);
        let unpacked = unpack_3bit_slice(&packed, 64).unwrap();
        assert_eq!(values, unpacked);
    }

    #[test]
    fn roundtrip_slice_4096() {
        let values: Vec<u8> = (0..4096u16).map(|x| (x % 8) as u8).collect();
        let packed = pack_3bit_slice(&values).unwrap();
        let unpacked = unpack_3bit_slice(&packed, 4096).unwrap();
        assert_eq!(values, unpacked);
    }

    #[test]
    fn unaligned_slice_errors() {
        let values = vec![0u8; 7];
        assert!(pack_3bit_slice(&values).is_err());
        let packed = vec![0u8; 6];
        assert!(unpack_3bit_slice(&packed, 7).is_err());
    }

    #[test]
    fn packed_slice_too_short() {
        let packed = vec![0u8; 3];
        assert!(unpack_3bit_slice(&packed, 16).is_err());
    }
}
