# PolarQuant

PolarQuant is the first phase: geometric rotation that spreads
information uniformly before quantization.

## Mathematics

```text
y = x · R    where Rᵀ · R = I
```

Properties: norm preserved, dot product preserved.

## Strategies

| Strategy | Speed | Memory | Best For |
|----------|-------|--------|----------|
| QR | O(d²) | O(d²) | Small heads (≤64) |
| Householder | O(kd) | O(kd) | Medium heads |
| Hadamard | O(d log d) | O(d) | Large heads (≥128) |
