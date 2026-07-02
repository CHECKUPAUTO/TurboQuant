//! Property-style round-trip tests for 3-bit packing/unpacking.
//!
//! Referenced from `BUGS_FIXED.md` (Bugs 1 & 2 of the Python port).

// Test-only: `vec!` literals keep the error-case assertions readable.
#![allow(clippy::useless_vec)]

use proptest::prelude::*;
use turboquant_core::bitpack::{pack_3bit, pack_3bit_slice, unpack_3bit, unpack_3bit_slice};

proptest! {
    /// Round-trip over arbitrary 8-value groups.
    #[test]
    fn single_group_roundtrip(values in prop::array::uniform8(0u8..8)) {
        let mut packed = [0u8; 3];
        pack_3bit(&values, &mut packed);
        let mut unpacked = [0u8; 8];
        unpack_3bit(&packed, &mut unpacked);
        prop_assert_eq!(values, unpacked);
    }

    /// Round-trip over slices of many sizes (multiples of 8, up to 4096).
    #[test]
    fn slice_roundtrip(
        values in (0usize..512)
            .prop_flat_map(|groups| prop::collection::vec(0u8..8, groups * 8))
    ) {
        let packed = pack_3bit_slice(&values).unwrap();
        prop_assert_eq!(packed.len(), values.len() * 3 / 8);
        let unpacked = unpack_3bit_slice(&packed, values.len()).unwrap();
        prop_assert_eq!(values, unpacked);
    }

    /// High bits beyond the 3-bit payload must be masked, not corrupt
    /// neighbouring values.
    #[test]
    fn high_bits_are_masked(values in prop::array::uniform8(0u8..=255)) {
        let mut packed = [0u8; 3];
        pack_3bit(&values, &mut packed);
        let mut unpacked = [0u8; 8];
        unpack_3bit(&packed, &mut unpacked);
        let expected: Vec<u8> = values.iter().map(|v| v & 0x07).collect();
        prop_assert_eq!(&expected[..], &unpacked[..]);
    }
}

#[test]
fn unaligned_lengths_error() {
    for n in [1usize, 7, 9, 63, 65] {
        assert!(
            pack_3bit_slice(&vec![0u8; n]).is_err(),
            "len {n} must error"
        );
        assert!(unpack_3bit_slice(&vec![0u8; n * 3 / 8 + 3], n).is_err());
    }
}

#[test]
fn empty_slice_roundtrip() {
    let packed = pack_3bit_slice(&[]).unwrap();
    assert!(packed.is_empty());
    let unpacked = unpack_3bit_slice(&packed, 0).unwrap();
    assert!(unpacked.is_empty());
}

#[test]
fn truncated_packed_data_errors() {
    let values: Vec<u8> = (0..64).map(|x| x % 8).collect();
    let packed = pack_3bit_slice(&values).unwrap();
    assert!(unpack_3bit_slice(&packed[..packed.len() - 1], 64).is_err());
}
