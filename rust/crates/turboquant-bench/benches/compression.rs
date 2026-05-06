use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::Rng;
use turboquant_core::bitpack::{pack_3bit_slice, unpack_3bit_slice};

fn bench_pack_3bit(c: &mut Criterion) {
    let mut rng = rand::thread_rng();
    let sizes = [64, 256, 1024, 4096];

    for &n in sizes {
        let values: Vec<u8> = (0..n).map(|_| rng.gen_range(0..8u8)).collect();
        c.bench_function(&format!("pack_3bit_{n}"), |b| {
            b.iter(|| pack_3bit_slice(black_box(&values)).unwrap());
        });

        let packed = pack_3bit_slice(&values).unwrap();
        c.bench_function(&format!("unpack_3bit_{n}"), |b| {
            b.iter(|| unpack_3bit_slice(black_box(&packed), n).unwrap());
        });
    }
}

criterion_group!(benches, bench_pack_3bit);
criterion_main!(benches);
