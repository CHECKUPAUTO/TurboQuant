//! PolarQuant: geometric rotation for KV cache compression.

use ndarray::ArrayViewMut2;
use crate::rotation::Rotation;

/// PolarQuant applies an orthogonal rotation before quantization.
///
/// This spreads information uniformly across dimensions,
/// making quantization error evenly distributed.
#[derive(Clone, Debug)]
pub struct PolarQuant<R: Rotation> {
    rotation: R,
}

impl<R: Rotation> PolarQuant<R> {
    /// Create a new PolarQuant with the given rotation strategy.
    #[must_use]
    pub fn new(rotation: R) -> Self {
        Self { rotation }
    }

    /// Apply forward rotation: `y = x · R`.
    pub fn forward(&self, x: &mut ArrayViewMut2<f32>) {
        self.rotation.forward(x);
    }

    /// Apply inverse rotation: `x = y · Rᵀ`.
    pub fn inverse(&self, y: &mut ArrayViewMut2<f32>) {
        self.rotation.inverse(y);
    }

    /// Return the dimension.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.rotation.dim()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rotation::QrRotation;
    use ndarray::Array2;

    #[test]
    fn test_polar_roundtrip() {
        let rot = QrRotation::new(64, Some(42));
        let pq = PolarQuant::new(rot);
        let x = Array2::from_shape_fn((16, 64), |_| rand::random::<f32>() * 2.0 - 1.0);
        let mut y = x.clone();
        pq.forward(&mut y.view_mut());
        pq.inverse(&mut y.view_mut());

        for i in 0..16 {
            for j in 0..64 {
                assert!((y[[i, j]] - x[[i, j]]).abs() < 1e-4);
            }
        }
    }
}
