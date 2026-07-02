//! Integration tests: GGUF container round-trip and the `TurboQuant`
//! compress/decompress pipeline on a synthetic model.

// Test-data generation casts (i32->f32, u32->i32) are intentional.
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_wrap)]

use turboquant_gguf::turbo::{self, TurboOptions, KEY_BITS, KEY_BLOCK_SIZE, KEY_FORMAT_VERSION};
use turboquant_gguf::{GgmlType, GgufParser, GgufValue, GgufValueType, GgufWriter};

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn f16_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|v| half::f16::from_f32(*v).to_le_bytes())
        .collect()
}

/// Build a small synthetic model: metadata of varied types + 3 tensors.
fn build_synthetic() -> (GgufWriter, Vec<f32>, Vec<f32>, Vec<u8>) {
    let mut w = GgufWriter::new();
    w.add_metadata("general.name", GgufValue::String("tiny-test".into()));
    w.add_metadata("general.file_type", GgufValue::U32(0));
    w.add_metadata("test.flag", GgufValue::Bool(true));
    w.add_metadata("test.u8", GgufValue::U8(7));
    w.add_metadata("test.i16", GgufValue::I16(-12));
    w.add_metadata("test.i64", GgufValue::I64(-1_234_567_890_123));
    w.add_metadata("test.f32", GgufValue::F32(0.25));
    w.add_metadata("test.f64", GgufValue::F64(1.5e300));
    w.add_metadata(
        "test.arr.u32",
        GgufValue::Array(
            GgufValueType::U32,
            vec![GgufValue::U32(1), GgufValue::U32(2), GgufValue::U32(3)],
        ),
    );
    w.add_metadata(
        "test.arr.str",
        GgufValue::Array(
            GgufValueType::String,
            vec![
                GgufValue::String("alpha".into()),
                GgufValue::String("beta".into()),
            ],
        ),
    );

    // Smooth F32 tensor (2-D).
    let t0: Vec<f32> = (0..2048).map(|i| (i as f32 * 0.01).sin() * 3.0).collect();
    w.add_tensor(
        "blk.0.attn_k.weight",
        vec![64, 32],
        GgmlType::F32,
        f32_bytes(&t0),
    )
    .unwrap();

    // Smooth F16 tensor (1-D, length not a multiple of the block size).
    let t1: Vec<f32> = (0..100).map(|i| (i as f32 * 0.05).cos()).collect();
    w.add_tensor(
        "blk.0.attn_v.weight",
        vec![100],
        GgmlType::F16,
        f16_bytes(&t1),
    )
    .unwrap();

    // Non-float tensor: passes through compression unchanged.
    let t2: Vec<u8> = (0..64u32)
        .flat_map(|i| (i as i32 - 32).to_le_bytes())
        .collect();
    w.add_tensor("token_ids", vec![64], GgmlType::I32, t2.clone())
        .unwrap();

    (w, t0, t1, t2)
}

#[test]
fn container_roundtrip_preserves_everything() {
    let (w, t0, t1, t2) = build_synthetic();
    let bytes = w.to_bytes().unwrap();

    let parsed = GgufParser::parse(bytes).unwrap();
    assert_eq!(parsed.header.version, 3);
    assert_eq!(parsed.header.tensor_count, 3);
    assert_eq!(parsed.header.metadata_kv_count, 10);
    assert_eq!(parsed.alignment, 32);

    // Metadata values survive with exact types.
    assert_eq!(
        parsed.metadata_value("general.name"),
        Some(&GgufValue::String("tiny-test".into()))
    );
    assert_eq!(parsed.metadata_value("test.u8"), Some(&GgufValue::U8(7)));
    assert_eq!(
        parsed.metadata_value("test.i16"),
        Some(&GgufValue::I16(-12))
    );
    assert_eq!(
        parsed.metadata_value("test.i64"),
        Some(&GgufValue::I64(-1_234_567_890_123))
    );
    assert_eq!(
        parsed.metadata_value("test.f32"),
        Some(&GgufValue::F32(0.25))
    );
    assert_eq!(
        parsed.metadata_value("test.f64"),
        Some(&GgufValue::F64(1.5e300))
    );
    assert_eq!(
        parsed.metadata_value("test.flag"),
        Some(&GgufValue::Bool(true))
    );
    assert_eq!(
        parsed.metadata_value("test.arr.u32"),
        Some(&GgufValue::Array(
            GgufValueType::U32,
            vec![GgufValue::U32(1), GgufValue::U32(2), GgufValue::U32(3)],
        ))
    );
    assert_eq!(
        parsed.metadata_value("test.arr.str"),
        Some(&GgufValue::Array(
            GgufValueType::String,
            vec![
                GgufValue::String("alpha".into()),
                GgufValue::String("beta".into()),
            ],
        ))
    );

    // Tensor infos.
    let i0 = parsed.tensor("blk.0.attn_k.weight").unwrap();
    assert_eq!(i0.dims, vec![64, 32]);
    assert_eq!(i0.ggml_type, GgmlType::F32);
    let i1 = parsed.tensor("blk.0.attn_v.weight").unwrap();
    assert_eq!(i1.ggml_type, GgmlType::F16);
    let i2 = parsed.tensor("token_ids").unwrap();
    assert_eq!(i2.ggml_type, GgmlType::I32);

    // Offsets are aligned.
    for t in &parsed.tensors {
        assert_eq!(
            t.offset % parsed.alignment,
            0,
            "tensor {} unaligned",
            t.name
        );
    }

    // Tensor data decodes back exactly.
    assert_eq!(parsed.tensor_f32(i0).unwrap(), t0);
    let t1_via_f16: Vec<f32> = t1
        .iter()
        .map(|&v| half::f16::from_f32(v).to_f32())
        .collect();
    assert_eq!(parsed.tensor_f32(i1).unwrap(), t1_via_f16);
    assert_eq!(parsed.tensor_data(i2).unwrap(), t2.as_slice());
}

#[test]
fn write_parse_write_is_stable() {
    let (w, _, _, _) = build_synthetic();
    let bytes = w.to_bytes().unwrap();
    let parsed = GgufParser::parse(bytes.clone()).unwrap();

    // Rebuild from the parsed representation; output must be identical.
    let mut w2 = GgufWriter::new();
    for (k, v) in &parsed.metadata {
        w2.add_metadata(k.clone(), v.clone());
    }
    for t in &parsed.tensors {
        w2.add_tensor(
            t.name.clone(),
            t.dims.clone(),
            t.ggml_type,
            parsed.tensor_data(t).unwrap().to_vec(),
        )
        .unwrap();
    }
    assert_eq!(w2.to_bytes().unwrap(), bytes);
}

#[test]
fn custom_alignment_roundtrip() {
    let mut w = GgufWriter::new();
    w.set_alignment(64).unwrap();
    w.add_tensor("t", vec![8], GgmlType::F32, f32_bytes(&[1.0; 8]))
        .unwrap();
    let parsed = GgufParser::parse(w.to_bytes().unwrap()).unwrap();
    assert_eq!(parsed.alignment, 64);
    assert_eq!(parsed.data_start % 64, 0);
    assert_eq!(
        parsed.tensor_f32(parsed.tensor("t").unwrap()).unwrap(),
        vec![1.0; 8]
    );
}

#[test]
fn parser_rejects_garbage() {
    assert!(GgufParser::parse(b"NOTG".to_vec()).is_err());
    assert!(GgufParser::parse(b"GGUF\x01\x00\x00\x00".to_vec()).is_err());
    // Valid header but truncated body.
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"GGUF");
    bytes.extend_from_slice(&3u32.to_le_bytes());
    bytes.extend_from_slice(&1u64.to_le_bytes()); // 1 tensor
    bytes.extend_from_slice(&0u64.to_le_bytes()); // 0 metadata
    assert!(GgufParser::parse(bytes).is_err());
}

#[test]
fn turbo_compress_roundtrip_snr() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("model.gguf");
    let output = dir.path().join("model-turbo3.gguf");

    let (w, t0, t1, t2) = build_synthetic();
    w.write_to_file(&input).unwrap();
    let input_size = std::fs::metadata(&input).unwrap().len();

    let stats = turbo::compress_file(&input, &output, &TurboOptions::default()).unwrap();
    assert_eq!(stats.tensors_compressed, 2);
    assert_eq!(stats.tensors_passthrough, 1);

    let parsed = GgufParser::parse_file(&output).unwrap();
    assert!(turbo::is_turbo_compressed(&parsed));
    assert_eq!(
        parsed.metadata_value(KEY_FORMAT_VERSION),
        Some(&GgufValue::U32(1))
    );
    assert_eq!(parsed.metadata_value(KEY_BITS), Some(&GgufValue::U8(3)));
    assert_eq!(
        parsed.metadata_value(KEY_BLOCK_SIZE),
        Some(&GgufValue::U32(64))
    );
    // Original metadata carried through.
    assert_eq!(
        parsed.metadata_value("general.name"),
        Some(&GgufValue::String("tiny-test".into()))
    );

    let names = turbo::turbo_tensor_names(&parsed);
    assert_eq!(names.len(), 2);

    // F32 tensor: reconstruction must be plausible on smooth data.
    let recon0 = turbo::decompress_tensor(&parsed, "blk.0.attn_k.weight").unwrap();
    assert_eq!(recon0.len(), t0.len());
    let snr0 = turbo::snr_db(&t0, &recon0);
    assert!(snr0 > 10.0, "F32 tensor SNR too low: {snr0:.2} dB");

    // F16 tensor with a partial trailing block.
    let recon1 = turbo::decompress_tensor(&parsed, "blk.0.attn_v.weight").unwrap();
    assert_eq!(recon1.len(), t1.len());
    let snr1 = turbo::snr_db(&t1, &recon1);
    assert!(snr1 > 10.0, "F16 tensor SNR too low: {snr1:.2} dB");

    // Non-float tensor passed through byte-identical.
    let i2 = parsed.tensor("token_ids").unwrap();
    assert_eq!(i2.ggml_type, GgmlType::I32);
    assert_eq!(parsed.tensor_data(i2).unwrap(), t2.as_slice());

    // 3-bit + 1-bit correction + scales should shrink float data
    // substantially (F32 payload goes 32 -> ~4.5 bits/value).
    let output_size = std::fs::metadata(&output).unwrap().len();
    assert!(
        output_size < input_size / 2,
        "expected >2x total shrink, got {input_size} -> {output_size}"
    );

    // Double compression is refused.
    let err = turbo::compress_file(&output, dir.path().join("x.gguf"), &TurboOptions::default());
    assert!(err.is_err());
}

#[test]
fn turbo_rejects_unsupported_bits() {
    let (w, _, _, _) = build_synthetic();
    let parsed = GgufParser::parse(w.to_bytes().unwrap()).unwrap();
    let opts = TurboOptions {
        bits: 4,
        ..Default::default()
    };
    assert!(turbo::compress(&parsed, &opts).is_err());
}
