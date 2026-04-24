# GPU Neural Field Simulation

Wilson-Cowan neural field equations on a 2D cortical sheet, solved with
FFT-based convolution on the GPU. Under the right parameters the system
spontaneously produces the geometric hallucination patterns (spirals,
tunnels, lattices) derived by Ermentrout & Cowan (1979) from the same
equations.

COE 379 final project, Spring 2026 -- Will Suan (whs726 / wsuan).

## What it does

Solves the two-population Wilson-Cowan equations on a periodic 2D grid:

```
tau_E dE/dt = -E + S( w_EE (K_E * E) - w_IE (K_I * I) + P,  beta_E, theta_E )
tau_I dI/dt = -I + S( w_EI (K_E * E) - w_II (K_I * I) + Q,  beta_I, theta_I )
```

- `E`, `I`: excitatory and inhibitory activity fields
- `K_E`, `K_I`: short-range / long-range Gaussian coupling kernels
- `S`: logistic sigmoid
- Convolution via cuFFT (R2C / C2R); Gaussian transforms are analytic in k-space
- Time integration: forward Euler

Per step: 2 R2C + 2 C2R FFTs + a handful of pointwise kernels. On a single
Quadro RTX 5000, a 1024x1024 run does ~30-80 steps/sec depending on stride.

## Layout

```
project/
  CMakeLists.txt
  src/
    main.cu          # driver, time loop, cuFFT setup
    kernels.cuh      # device kernels (init, Gaussian hat, multiply, update, colormap)
    io.hpp / io.cpp  # PPM writer, preset table, argument parser
  scripts/
    build_local.sh   # configure + build (interactive / idev)
    job_single.sh    # SLURM: one preset on one GPU
    job_sweep.sh     # SLURM: four presets in parallel, one per GPU
    make_video.sh    # ffmpeg wrapper: PPM frames -> mp4
  configs/           # (room for extended parameter files)
  proposal.md        # original project proposal
  README.md          # this file
```

## Building on Frontera

From a login node:

```
module load cuda
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Or: `scripts/build_local.sh`.

`CMAKE_CUDA_ARCHITECTURES` defaults to `75` (Quadro RTX 5000). Override
with `-DCMAKE_CUDA_ARCHITECTURES=86` for an A100/A30 node elsewhere.

## Running

### Single run, one GPU

```
sbatch scripts/job_single.sh spirals
```

Positional arg is the preset (`spots`, `stripes`, `spirals`, `waves`, `default`).
Override grid size and step count via env vars:

```
NX=2048 NY=2048 NSTEPS=8000 sbatch scripts/job_single.sh spirals
```

### Parameter sweep, four GPUs in parallel

```
sbatch scripts/job_sweep.sh
```

Launches four presets concurrently on the four GPUs of a single rtx-dev
node via `CUDA_VISIBLE_DEVICES`. Each writes its own log to `logs/` and its
frames to `frames/<preset>/`.

### Video

After a run:

```
scripts/make_video.sh frames/spirals
```

Produces `frames/spirals.mp4`. Needs `ffmpeg` -- on Frontera,
`module load ffmpeg` first.

## Flags

`neural_field` accepts `--key=value` arguments. All struct fields in
`io.hpp::WCParams` are overridable:

```
--preset=<name>            one of: default, spots, stripes, spirals, waves
--nx=<int>  --ny=<int>     grid size (default 1024)
--nsteps=<int>             total timesteps (default 4000)
--frame-stride=<int>       write one frame every N steps (default 20)
--dt=<float>               timestep (default 0.1)
--sigma-e=<float>          excitatory kernel width in grid units
--sigma-i=<float>          inhibitory kernel width in grid units
--w-ee --w-ie --w-ei --w-ii   coupling weights
--p --q                    constant external drive
--beta-e --theta-e         sigmoid slope / threshold (excitatory)
--beta-i --theta-i         sigmoid slope / threshold (inhibitory)
--tau-e --tau-i            time constants
--seed=<uint>              PRNG seed for IC noise
--out=<dir>                output directory for PPM frames
```

## Interactive testing (idev)

```
idev -p rtx-dev -N 1 -n 1 -t 00:30:00
module load cuda
scripts/build_local.sh
./build/neural_field --preset=spirals --nsteps=2000 --out=frames/test
```

## References

- H. R. Wilson and J. D. Cowan, "Excitatory and inhibitory interactions in
  localized populations of model neurons," *Biophysical Journal* 12, 1972.
- G. B. Ermentrout and J. D. Cowan, "A mathematical theory of visual
  hallucination patterns," *Biological Cybernetics* 34, 1979.
- P. C. Bressloff et al., "Geometric visual hallucinations, Euclidean
  symmetry and the functional architecture of striate cortex,"
  *Phil. Trans. Roy. Soc. B* 356, 2001.
