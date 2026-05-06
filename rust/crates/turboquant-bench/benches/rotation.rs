use criterion::{criterion_group, criterion_main, Criterion};
use ndarray::Array2;
use turboquant_core::rotation::{FastHadamardRotation, HouseholderRotation, QrRotation, Rotation};

fn bench_rotation_init(c: &mut Criterion) {
    let dims = [64, 128, 256];

    for &dim in &dims {
        c.bench_function(&format!("qr_init_{dim}"), |b| {
            b.iter(|| QrRotation::new(dim, Some(42)));
        });

        c.bench_function(&format!("householder_init_{dim}"), |b| {
            b.iter(|| HouseholderRotation::new(dim, 16, Some(42)));
        });

        if dim.is_power_of_two() {
            c.bench_function(&format!("hadamard_init_{dim}"), |b| {
                b.iter(|| FastHadamardRotation::new(dim, Some(42)));
            });
        }
    }
}

fn bench_rotation_apply(c: &mut Criterion) {
    let dims = [64, 128, 256];
    let batch = 16;

    for &dim in &dims {
        let qr = QrRotation::new(dim, Some(42));
        let hh = HouseholderRotation::new(dim, 16, Some(42));
        let x = Array2::from_shape_fn((batch, dim), |_| rand::random::<f32>() * 2.0 - 1.0);

        c.bench_function(&format!("qr_forward_{dim}"), |b| {
            b.iter_batched(
                || x.clone(),
                |mut x| qr.forward(&mut x.view_mut()),
                criterion::BatchSize::SmallInput,
            );
        });

        c.bench_function(&format!("householder_forward_{dim}"), |b| {
            b.iter_batched(
                || x.clone(),
                |mut x| hh.forward(&mut x.view_mut()),
                criterion::BatchSize::SmallInput,
            );
        });

        if dim.is_power_of_two() {
            let fh = FastHadamardRotation::new(dim, Some(42));
            c.bench_function(&format!("hadamard_forward_{dim}"), |b| {
                b.iter_batched(
                    || x.clone(),
                    |mut x| fh.forward(&mut x.view_mut()),
                    criterion::BatchSize::SmallInput,
                );
            });
        }
    }
}

criterion_group!(init, bench_rotation_init);
criterion_group!(apply, bench_rotation_apply);
criterion_main!(init, apply);
