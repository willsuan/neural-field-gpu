# Project Proposal — GPU Neural Field Simulation

**Student:** Will Suan (whs726 / wsuan)
**Platform:** TACC Frontera RTX (4 × NVIDIA Quadro RTX 5000 per node)

## Proposal

I will implement a large-scale simulation of the Wilson-Cowan neural field
equations on a 2D cortical sheet, targeting multi-GPU execution on a Frontera
RTX node. The model couples two scalar fields — excitatory E(x,t) and
inhibitory I(x,t) — via a Mexican hat convolution kernel:

```
τ_E · dE/dt = −E + S( w_EE·(φ_EE*E) − w_IE·(φ_IE·I) + P )
τ_I · dI/dt = −I + S( w_EI·(φ_EI*E) − w_II·(φ_II·I) + Q )
```

Under the right parameters, this spontaneously produces the geometric
hallucination patterns (spirals, tunnels, lattices, cobwebs) first derived by
Ermentrout & Cowan (1979) from the same equations. The output — color-mapped
frames stitched into video — is both scientifically meaningful and visually
striking.

## HPC relevance

The dominant cost per timestep is the 2D convolution with the coupling kernel.
Direct convolution is O(N² · K²); FFT-based convolution via cuFFT reduces this
to O(N² log N) and is the natural GPU workload. This gives a direct performance
comparison between two parallel strategies — a concrete design decision that
drives the writeup.

## Parallel design

- **Single-GPU kernel:** pointwise sigmoid + Euler/RK4 update are
  embarrassingly parallel; convolution via cuFFT.
- **Multi-GPU:** partition the 2D grid across 4 GPUs, halo exchange via
  `cudaMemcpyPeer` for the finite-support kernel. 
- **Parameter sweep:** different parameter sets produce different pattern
  classes (spots, spirals, traveling waves). Each configuration runs
  independently on a separate GPU — embarrassingly parallel across the
  phase diagram.

## Evaluation

- **Correctness:** reproduce the published form-constant phase diagram
  (compare pattern types to Ermentrout & Cowan 1979, Fig. 3).
- **Performance:** cuFFT convolution vs. direct convolution; roofline
  analysis showing the bandwidth-bound regime.
- **Scaling:** strong scaling on 1→4 RTX GPUs for a fixed large grid.
- **Parameter sweep:** tile the phase diagram in parallel across GPUs;
  assemble a poster-style grid of output patterns.

## Scope

~700–900 LOC C++ / CUDA, built with CMake like hw7–hw9. Single-GPU correct
version first, then multi-GPU, then parameter sweep. Roughly six weeks.
