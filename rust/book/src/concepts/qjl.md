# QJL — Quantized Johnson-Lindenstrauss

3-bit quantization with 1-bit residual correction, guaranteeing
dot product preservation within the JL bound.

## Scale Modes

- **AbsMax**: Simple but outlier-sensitive
- **Percentile**: Robust to outliers (p99.5)
- **Adaptive**: Derived from variance theory
- **Fixed**: User-specified
