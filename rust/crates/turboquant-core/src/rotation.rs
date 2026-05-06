//! Orthogonal rotation matrices for PolarQuant.
//!
//! Three strategies:
//! - QR decomposition (baseline, O(d²) apply)
//! - Householder reflectors (O(k·d) apply)
//! - Fast Hadamard transform (O(d log d) apply)

use ndarray::{Array1, Array2, ArrayView1, ArrayViewMut2};
use rand::Rng;
use rand_distr::StandardNormal;
use serde::{Deserialize, Serialize};
use tracing::debug;

/// Trait for orthogonal rotation strategies.
pub trait Rotation: Send + Sync {
    /// Apply forward rotation: `y = x · R`.
    fn forward(&self, x: &mut ArrayViewMut2<f32>);

    /// Apply inverse rotation: `x = y · Rᵀ`.
    fn inverse(&self, y: &mut ArrayViewMut2<f32>);

    /// Return the dimension of the rotation.
    fn dim(&self) -> usize;
}

/// QR-based orthogonal rotation (baseline).
///
/// Stores the full `d × d` matrix. Apply is O(d²).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QrRotation {
    /// The orthogonal rotation matrix (d × d).
    pub matrix: Array2<f32>,
}

impl QrRotation {
    /// Create a new random QR rotation.
    ///
    /// # Examples
    ///
    /// ```
    /// use turboquant_core::rotation::QrRotation;
    ///
    /// let rot = QrRotation::new(64);
    /// assert_eq!(rot.dim(), 64);
    /// ```
    #[allow(clippy::cast_possible_truncation)]
    pub fn new(dim: usize, seed: Option<u64>) -> Self {
        let mut rng: rand::rngs::StdRng = if let Some(s) = seed {
            rand::SeedableRng::seed_from_u64(s)
        } else {
            rand::SeedableRng::from_entropy()
        };

        let h: Array2<f32> = Array2::from_shape_fn((dim, dim), |_| {
            rng.sample(StandardNormal)
        });

        // QR decomposition via Gram-Schmidt
        let mut q = Array2::<f32>::zeros((dim, dim));
        for j in 0..dim {
            let mut v = h.column(j).to_owned();
            for i in 0..j {
                let qi = q.column(i);
                let dot = v.dot(&qi);
                v = v - &(qi.mapv(|x| x * dot));
            }
            let norm = v.dot(&v).sqrt();
            if norm > 1e-10 {
                v.mapv_inplace(|x| x / norm);
            }
            q.column_mut(j).assign(&v);
        }
        debug!("Created QrRotation(dim={})", dim);
        Self { matrix: q }
    }

    /// Verify orthogonality within tolerance.
    #[must_use]
    pub fn is_orthogonal(&self, tolerance: f32) -> bool {
        let dim = self.matrix.nrows();
        let identity = Array2::<f32>::eye(dim);
        let should_be_id = self.matrix.t().dot(&self.matrix);
        let diff = &should_be_id - &identity;
        diff.mapv(|x| x * x).sum().sqrt() < tolerance * (dim as f32).sqrt()
    }
}

impl Rotation for QrRotation {
    fn forward(&self, x: &mut ArrayViewMut2<f32>) {
        // y = x · R (right-multiply with row vectors)
        let x_copy = x.to_owned();
        x.assign(&x_copy.dot(&self.matrix));
    }

    fn inverse(&self, y: &mut ArrayViewMut2<f32>) {
        // x = y · Rᵀ
        let y_copy = y.to_owned();
        y.assign(&y_copy.dot(&self.matrix.t()));
    }

    fn dim(&self) -> usize {
        self.matrix.nrows()
    }
}

/// Householder-based rotation (k reflectors, O(k·d) apply).
///
/// More memory-efficient than QR for large d.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HouseholderRotation {
    /// Dimension of the rotation.
    dim: usize,
    /// Householder reflector vectors.
    reflectors: Vec<Array1<f32>>,
}

impl HouseholderRotation {
    /// Create a new Householder rotation with `k` random reflectors.
    ///
    /// # Examples
    ///
    /// ```
    /// use turboquant_core::rotation::HouseholderRotation;
    ///
    /// let rot = HouseholderRotation::new(128, 16, Some(42));
    /// assert_eq!(rot.dim(), 128);
    /// ```
    pub fn new(dim: usize, num_reflectors: usize, seed: Option<u64>) -> Self {
        let mut rng: rand::rngs::StdRng = if let Some(s) = seed {
            rand::SeedableRng::seed_from_u64(s)
        } else {
            rand::SeedableRng::from_entropy()
        };

        let reflectors: Vec<Array1<f32>> = (0..num_reflectors)
            .map(|_| {
                Array1::from_shape_fn(dim, |_| rng.sample(StandardNormal))
            })
            .collect();

        debug!(
            "Created HouseholderRotation(dim={}, k={})",
            dim, num_reflectors
        );
        Self { dim, reflectors }
    }

    /// Apply a Householder reflector: `v -> v - 2·(v·u)/(u·u)·u`.
    #[allow(clippy::cast_precision_loss)]
    fn apply_reflector(x: &mut ArrayViewMut2<f32>, u: &ArrayView1<f32>) {
        let uu = u.dot(u);
        if uu < 1e-12 {
            return;
        }
        let nrows = x.nrows();
        for i in 0..nrows {
            let mut row = x.row_mut(i);
            let dot = row.dot(u);
            let factor = 2.0 * dot / uu;
            row.scaled_add(-factor, u);
        }
    }
}

impl Rotation for HouseholderRotation {
    fn forward(&self, x: &mut ArrayViewMut2<f32>) {
        for u in &self.reflectors {
            Self::apply_reflector(x, &u.view());
        }
    }

    fn inverse(&self, y: &mut ArrayViewMut2<f32>) {
        // Householder is its own inverse, applied in reverse order.
        for u in self.reflectors.iter().rev() {
            Self::apply_reflector(y, &u.view());
        }
    }

    fn dim(&self) -> usize {
        self.dim
    }
}

/// Fast Hadamard-based rotation (O(d log d) apply).
///
/// Uses the Walsh-Hadamard transform + random sign flips.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FastHadamardRotation {
    /// Dimension (must be a power of two).
    dim: usize,
    /// Random sign flips.
    signs: Vec<i8>,
}

impl FastHadamardRotation {
    /// Create a new Hadamard rotation.
    ///
    /// # Panics
    ///
    /// Panics if `dim` is not a power of two.
    pub fn new(dim: usize, seed: Option<u64>) -> Self {
        assert!(
            dim.is_power_of_two(),
            "dim must be a power of two, got {dim}"
        );

        let mut rng: rand::rngs::StdRng = if let Some(s) = seed {
            rand::SeedableRng::seed_from_u64(s)
        } else {
            rand::SeedableRng::from_entropy()
        };

        let signs: Vec<i8> = (0..dim)
            .map(|_| if rng.gen_bool(0.5) { 1 } else { -1 })
            .collect();

        debug!("Created FastHadamardRotation(dim={})", dim);
        Self { dim, signs }
    }

    /// In-place Fast Hadamard Transform (Walsh-Hadamard).
    #[allow(clippy::cast_possible_truncation)]
    fn fht_1d(v: &mut [f32]) {
        let n = v.len();
        let mut step = 1;
        while step < n {
            for i in (0..n).step_by(step * 2) {
                for j in 0..step {
                    let a = v[i + j];
                    let b = v[i + j + step];
                    v[i + j] = a + b;
                    v[i + j + step] = a - b;
                }
            }
            step *= 2;
        }
    }
}

impl Rotation for FastHadamardRotation {
    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    fn forward(&self, x: &mut ArrayViewMut2<f32>) {
        let ncols = x.ncols();
        let nrows = x.nrows();
        let scale = 1.0 / (self.dim as f32).sqrt();

        for i in 0..nrows {
            let mut row: Vec<f32> = (0..ncols).map(|j| x[[i, j]]).collect();
            // Apply sign flips
            for (j, val) in row.iter_mut().enumerate() {
                *val *= f32::from(self.signs[j]);
            }
            // FHT
            Self::fht_1d(&mut row);
            // Scale
            for val in &mut row {
                *val *= scale;
            }
            // Write back
            for (j, val) in row.iter().enumerate() {
                x[[i, j]] = *val;
            }
        }
    }

    fn inverse(&self, y: &mut ArrayViewMut2<f32>) {
        // Hadamard is self-inverse (up to scale). Apply same transform
        // then apply signs again.
        let ncols = y.ncols();
        let nrows = y.nrows();
        let scale = 1.0 / (self.dim as f32).sqrt();

        for i in 0..nrows {
            let mut row: Vec<f32> = (0..ncols).map(|j| y[[i, j]]).collect();
            // Unscale
            for val in &mut row {
                *val /= scale;
            }
            // FHT (self-inverse up to scale)
            Self::fht_1d(&mut row);
            // Apply signs again
            for (j, val) in row.iter_mut().enumerate() {
                *val *= f32::from(self.signs[j]);
            }
            // Scale back
            for val in &mut row {
                *val *= scale;
            }
            for (j, val) in row.iter().enumerate() {
                y[[i, j]] = *val;
            }
        }
    }

    fn dim(&self) -> usize {
        self.dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn test_rotation_roundtrip<R: Rotation>(rot: &R) {
        let dim = rot.dim();
        let x_arr = Array2::from_shape_fn((16, dim), |_| rand::random::<f32>() * 2.0 - 1.0);
        let mut x = x_arr.clone();
        rot.forward(&mut x.view_mut());
        rot.inverse(&mut x.view_mut());
        for i in 0..16 {
            for j in 0..dim {
                assert!(
                    (x[[i, j]] - x_arr[[i, j]]).abs() < 1e-4,
                    "Roundtrip failed at [{i},{j}]: {} vs {}",
                    x[[i, j]],
                    x_arr[[i, j]]
                );
            }
        }
    }

    #[test]
    fn test_qr_roundtrip() {
        let rot = QrRotation::new(64, Some(42));
        test_rotation_roundtrip(&rot);
    }

    #[test]
    fn test_qr_orthogonality() {
        let rot = QrRotation::new(64, Some(42));
        assert!(rot.is_orthogonal(1e-4));
    }

    #[test]
    fn test_householder_roundtrip() {
        let rot = HouseholderRotation::new(128, 16, Some(42));
        test_rotation_roundtrip(&rot);
    }

    #[test]
    fn test_hadamard_roundtrip() {
        let rot = FastHadamardRotation::new(128, Some(42));
        test_rotation_roundtrip(&rot);
    }

    #[test]
    fn test_qr_preserves_norm() {
        let rot = QrRotation::new(64, Some(42));
        let x = Array2::from_shape_fn((1, 64), |_| rand::random::<f32>() * 2.0 - 1.0);
        let norm_before = x.dot(&x.t())[[0, 0]].sqrt();
        let mut y = x.clone();
        rot.forward(&mut y.view_mut());
        let norm_after = y.dot(&y.t())[[0, 0]].sqrt();
        assert!(
            (norm_before - norm_after).abs() < 1e-4,
            "Norm not preserved: {} vs {}",
            norm_before,
            norm_after
        );
    }
}
