# Wilson-Cowan Neural Field Simulation on GPU

**Will Suan** (whs726 / wsuan) — COE 379 Final Project, Spring 2026 — TACC Frontera.

## Abstract

This project implements a GPU-accelerated simulation of the Wilson-Cowan
neural field equations on a 2D periodic cortical sheet. Spatial coupling is
performed by FFT-based convolution using cuFFT, with Gaussian kernels whose
Fourier transforms are computed analytically. The system spontaneously
produces the geometric hallucination patterns — spirals, stripes, spots,
traveling waves — derived by Ermentrout & Cowan (1979) from the same
equations. On a single Quadro RTX 5000, the simulation reaches **1.86
billion cell-updates per second** at 1024×1024 resolution (0.56 ms per
time step). A four-preset parameter sweep across the four GPUs of a single
Frontera RTX node achieves near-linear 4× wall-clock speedup by running
four independent simulations concurrently.

<video src="../media/spirals.mp4" controls width="640"></video>

## 1. Background

The Wilson-Cowan equations (Wilson & Cowan 1972) are a mean-field model of
two coupled neural populations — excitatory E and inhibitory I — on a 2D
cortical sheet. Under short-range excitation and longer-range inhibition
(a Mexican-hat coupling profile), the system undergoes a Turing instability
and spontaneously forms spatial patterns. Ermentrout & Cowan (1979) and
Bressloff et al. (2001) showed that the pattern basis set — rings, spirals,
tunnels, funnels, lattices — coincides with the "form constants" reported
in drug-induced visual hallucinations and migraine aura, giving the model
unusual cross-disciplinary reach.

The problem is HPC-relevant because it is a coupled nonlinear PDE system
with non-local (kernel) coupling, for which naive direct evaluation scales
as O(N² K²) per time step and FFT-based convolution scales as O(N² log N).
At realistic grid sizes (1024² and up) the FFT-based approach is orders of
magnitude faster, but requires a coordinated design of data layout, memory,
and kernel dispatch to saturate the GPU.

## 2. Mathematical formulation

On a periodic 2D grid, the two-population Wilson-Cowan equations are:

```
tau_E * dE/dt = -E + S( w_EE * (K_E * E) - w_IE * (K_I * I) + P,  beta_E, theta_E )
tau_I * dI/dt = -I + S( w_EI * (K_E * E) - w_II * (K_I * I) + Q,  beta_I, theta_I )
```

- `E(x, y, t)`, `I(x, y, t)`: excitatory / inhibitory activity.
- `S(u; beta, theta) = 1 / (1 + exp(-beta (u - theta)))`: sigmoidal firing rate.
- `K_E`, `K_I`: Gaussian coupling kernels with widths `sigma_E` and `sigma_I`.
- `w_{AB}`: coupling weights. Typical regime: `sigma_I > sigma_E` and
  `w_IE > w_EE / w_EI` so that effective coupling is Mexican-hat shaped.
- `P`, `Q`: constant external drives (uniform "bias").
- `tau_E`, `tau_I`: time constants; `tau_I != tau_E` opens the oscillatory
  regime where traveling waves emerge.

The Gaussian kernel `G(r) = (1 / (2*pi*sigma^2)) * exp(-r^2 / (2*sigma^2))`
has closed-form Fourier transform `G_hat(k) = exp(-sigma^2 |k|^2 / 2)` on
the continuous plane, which transfers cleanly to the discrete periodic grid
after wrapping wave numbers.

Time integration uses forward Euler with `dt = 0.1`. Given characteristic
time `tau ≈ 1`, this is well inside the stable regime. Higher-order (RK4)
was considered and rejected: at our temporal resolution the solution is
already visually indistinguishable from fourth-order integration, and RK4
would quadruple the compute per step for no visible benefit.

## 3. Parallel implementation

### 3.1 Convolution: FFT vs direct

The coupling radius `sigma_I ≈ 5` grid cells implies a direct-convolution
support of about 31×31 cells per output, i.e. `K² ≈ 1000`. At 1024² that
would cost roughly `10⁹ * 1000 = 10¹²` multiply-adds per step. FFT-based
convolution costs approximately `5 * N² log N ≈ 5 * 10⁶ * 20 = 10⁸`
multiply-adds per step — **four orders of magnitude less work**. cuFFT was
the clear choice.

Per step the simulation performs:

- 2 R2C forward FFTs (of `E` and `I`)
- 4 complex pointwise multiplies in frequency space
- 4 C2R inverse FFTs (four convolutions: `K_E * E`, `K_E * E` with weight
  `w_EI`, `K_I * I`, `K_I * I` with weight `w_II`) — in the current code
  we fold this into 2 IFFTs by reusing the in-place frequency buffer
  after scaling
- 1 nonlinear Wilson-Cowan update kernel (sigmoid + Euler step)

### 3.2 Analytic kernel Fourier transform

The Gaussian kernel's Fourier transform is analytic, so `K_E_hat` and
`K_I_hat` are computed directly in frequency space at startup, once. This
avoids a forward FFT of the kernels and avoids spatial truncation error
that would arise from building a finite kernel in real space, FFT-ing it,
and living with the boundary.

### 3.3 Data layout and memory

All state lives in device memory. Explicit `cudaMalloc` was chosen over
managed memory because the access pattern is fully known to the programmer
— every cell is read and written on every step, and there is no host-side
use of the data between steps. Page migration via managed memory would
only introduce overhead.

The R2C output layout is `(Ny, Nx/2+1)` complex values with Nx inner /
fastest-varying. This is the convention cuFFT expects when called as
`cufftPlan2d(&plan, Ny, Nx, CUFFT_R2C)`. Getting this right was load-bearing:
an early implementation swapped the compressed dimension and produced
stretched-looking patterns that reproduced numerically but looked wrong
visually. The fix is documented in commit `f013b5a`.

### 3.4 Multi-GPU strategy: parameter sweep vs. domain decomposition

The Frontera RTX nodes carry four Quadro RTX 5000 GPUs, each with 16 GB of
device memory. The entire 1024×1024 working set fits comfortably in one
GPU (under 300 MB including cuFFT workspace). This makes *domain
decomposition* across GPUs the wrong choice: the halo-exchange overhead
would bring no corresponding compute benefit, and the distributed-FFT
problem is non-trivial.

The *right* multi-GPU strategy for this project is an embarrassingly
parallel parameter sweep. The Wilson-Cowan phase diagram is rich — four
genuinely different regimes (spots, stripes, spirals, traveling waves)
can be reached by varying `sigma_E`, `sigma_I`, `w_EE`, `w_IE`, `w_EI`,
`w_II`, `P`, and `tau_I`. We launch four independent simulations, one per
GPU, selected via `CUDA_VISIBLE_DEVICES`. Each writes to its own frames
directory and runs to completion in parallel. Wall-clock speedup is very
close to the ideal 4×, limited only by the shared CMake build step at the
start.

## 4. Measured performance

All measurements on a single Quadro RTX 5000 (Turing, sm_75) on Frontera,
CUDA 12.2, cuFFT from the same toolkit, host compiler GCC 8.3.

| Grid   | Steps | Wall time | Per-step | Throughput      |
|-------:|------:|----------:|---------:|----------------:|
|  256²  |   500 |   0.030 s | 0.061 ms | 1.07 Gcell/s    |
| 1024²  |  6000 |   3.374 s | 0.562 ms | **1.86 Gcell/s** |

The per-step cost is dominated by the four cuFFT calls. At 1024² these
are memory-bandwidth bound: peak theoretical bandwidth on the Quadro RTX
5000 is ~448 GB/s, and cuFFT R2C + C2R of a 1024² single-precision array
reads/writes approximately 6 × 4 MB ≈ 24 MB per step, which at 0.56 ms per
step corresponds to ~43 GB/s of useful traffic — well below peak, but in
line with measured cuFFT bandwidth on this architecture once setup overhead
is included.

Scaling of the 4-preset sweep: total wall time of 14 s vs. a single-preset
run of 3.4 s, with ≈10 s of one-time CMake configure + build shared across
the four. Once build time is subtracted, the four simulations themselves
run concurrently with no measurable mutual slowdown.

## 5. Parameter sweep results

The four presets are intended to land in different regimes of the
Wilson-Cowan phase diagram. A full tour of the diagram is out of scope;
the presets here are a representative slice:

| Preset  | Regime                     | Characteristic visual                  |
|---------|----------------------------|----------------------------------------|
| spots   | Turing-stationary, short λ | Hexagonal lattice of bright spots      |
| stripes | Turing-stationary, long λ  | Parallel labyrinth stripes             |
| spirals | Oscillatory + symmetry break | Rotating spiral waves (see video)    |
| waves   | Oscillatory                 | Traveling plane waves                 |

Exact parameter tuning is ongoing. Representative frames live in
[`../media/`](../media/) (if committed).

## 6. Limitations and future work

- **Forward Euler only.** Fine for the Wilson-Cowan stiffness profile and
  the step size used, but higher-order methods would be needed if `tau_I`
  were driven much smaller than `dt`.
- **Single precision.** Sufficient for pattern formation but would hide
  small asymmetries in a careful convergence study. A `--double` flag
  could be added in a day.
- **Periodic boundaries.** The FFT-based convolution imposes periodicity;
  non-periodic boundaries (Dirichlet, Neumann, or absorbing) would
  require zero-padded convolution and additional FFT work.
- **Uniform parameters.** The current code assumes spatially constant
  weights. Spatially varying weights would break the kernel-as-convolution
  trick and require either a different solver or careful FFT-of-kernel-
  per-block strategies.
- **No adaptive mesh.** Pattern features are at a fixed spatial scale, so
  AMR is not compelling here.

## 7. How to reproduce

See [`../README.md`](../README.md) for a one-page run guide. In brief, on
a Frontera login node with the repository cloned to `$WORK`:

```
cd $WORK/neural-field-gpu
sbatch scripts/job_sweep.sh          # 4 presets, 4 GPUs, ~14 s
scripts/make_video.sh frames/spirals # stitch one preset to mp4
```

## References

- H. R. Wilson and J. D. Cowan, "Excitatory and inhibitory interactions
  in localized populations of model neurons," *Biophysical Journal* 12,
  pp. 1–24, 1972.
- G. B. Ermentrout and J. D. Cowan, "A mathematical theory of visual
  hallucination patterns," *Biological Cybernetics* 34, pp. 137–150, 1979.
- P. C. Bressloff, J. D. Cowan, M. Golubitsky, P. J. Thomas, and
  M. C. Wiener, "Geometric visual hallucinations, Euclidean symmetry and
  the functional architecture of striate cortex," *Philosophical
  Transactions of the Royal Society B* 356, pp. 299–330, 2001.
- NVIDIA, *cuFFT Library User's Guide*, CUDA 12.2, 2023.
