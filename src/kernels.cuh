/*
 * kernels.cuh - Device kernels for Wilson-Cowan neural field simulation
 *
 * The main pieces:
 *   init_fields            -- random noise IC with optional central bump
 *   gaussian_kernel_hat    -- FT of a Gaussian coupling kernel, direct in k-space
 *   multiply_hat           -- pointwise complex multiply in Fourier domain
 *   wilson_cowan_update    -- Euler step of the two-field ODE system
 *   colormap_rgb           -- scalar field -> RGB for frame output
 */

#pragma once

#include <cuda_runtime.h>
#include <cufft.h>

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

__device__ __forceinline__ float sigmoid(float u, float beta, float theta) {
  return 1.0f / (1.0f + expf(-beta * (u - theta)));
}

// cheap xorshift PRNG, seeded per-thread
__device__ __forceinline__ unsigned int xorshift32(unsigned int x) {
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  return x;
}

// ---------------------------------------------------------------------------
// initialization
// ---------------------------------------------------------------------------

__global__ void init_fields_kernel(float *E, float *I, int Nx, int Ny,
                                   float noise_amp, float bump_amp,
                                   float bump_sigma, unsigned int seed) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= Nx || y >= Ny)
    return;
  const int idx = y * Nx + x;

  // two independent noise streams per cell
  unsigned int s1 = seed + idx * 2654435761u;
  unsigned int s2 = seed + idx * 40503u + 17u;
  s1 = xorshift32(s1);
  s2 = xorshift32(s2);
  const float nE = (float)s1 / (float)0xFFFFFFFFu - 0.5f;  // in [-0.5, 0.5]
  const float nI = (float)s2 / (float)0xFFFFFFFFu - 0.5f;

  // optional central bump to break symmetry and seed patterns
  const float cx = 0.5f * (float)Nx;
  const float cy = 0.5f * (float)Ny;
  const float rx = (float)x - cx;
  const float ry = (float)y - cy;
  const float r2 = rx * rx + ry * ry;
  const float bump = bump_amp * expf(-r2 / (2.0f * bump_sigma * bump_sigma));

  E[idx] = noise_amp * nE + bump;
  I[idx] = noise_amp * nI;
}

// ---------------------------------------------------------------------------
// Gaussian kernel in Fourier space
// ---------------------------------------------------------------------------
//
// For G(r) = (1/(2*pi*sigma^2)) * exp(-r^2 / (2*sigma^2)) (unit integral),
// the Fourier transform on a periodic grid is G_hat(k) = exp(-sigma^2 * |k|^2 / 2).
//
// cuFFT R2C on a real array of shape (Ny, Nx) (Ny outer / slowest, Nx inner /
// fastest) produces complex output of shape (Ny, Nxh) where Nxh = Nx/2 + 1.
// Only non-negative x frequencies are stored; y wraps the full range.
//   kx in [0, Nxh)          -> wx = 2*pi*kx / (Nx*dx)
//   ky in [0, Ny)           -> ky_eff = (ky <= Ny/2) ? ky : ky - Ny
//                              wy = 2*pi*ky_eff / (Ny*dy)

__global__ void gaussian_kernel_hat_kernel(cufftComplex *K_hat, int Nx, int Ny,
                                           float sigma, float dx, float dy) {
  const int kx = blockIdx.x * blockDim.x + threadIdx.x;   // 0..Nxh-1
  const int ky = blockIdx.y * blockDim.y + threadIdx.y;   // 0..Ny-1
  const int Nxh = Nx / 2 + 1;
  if (kx >= Nxh || ky >= Ny)
    return;

  const int ky_eff = (ky <= Ny / 2) ? ky : ky - Ny;
  const float wx = 2.0f * (float)M_PI * (float)kx / ((float)Nx * dx);
  const float wy = 2.0f * (float)M_PI * (float)ky_eff / ((float)Ny * dy);
  const float k2 = wx * wx + wy * wy;
  const float val = expf(-sigma * sigma * k2 * 0.5f);

  const int idx = ky * Nxh + kx;  // row-major (ky, kx)
  K_hat[idx].x = val;
  K_hat[idx].y = 0.0f;
}

// ---------------------------------------------------------------------------
// pointwise complex multiply in Fourier domain
// ---------------------------------------------------------------------------

__global__ void multiply_hat_kernel(cufftComplex *field_hat,
                                    const cufftComplex *K_hat, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  const cufftComplex a = field_hat[i];
  const cufftComplex b = K_hat[i];
  cufftComplex c;
  c.x = a.x * b.x - a.y * b.y;
  c.y = a.x * b.y + a.y * b.x;
  field_hat[i] = c;
}

// ---------------------------------------------------------------------------
// Wilson-Cowan Euler update
// ---------------------------------------------------------------------------
//
// tau_E dE/dt = -E + S( w_EE * (K_E * E)  - w_IE * (K_I * I) + P,  beta_E, theta_E )
// tau_I dI/dt = -I + S( w_EI * (K_E * E)  - w_II * (K_I * I) + Q,  beta_I, theta_I )
//
// Conv_E and Conv_I come in un-normalized from cuFFT (scaled by Nx*Ny); we
// divide by norm = 1/(Nx*Ny) here.

__global__ void wilson_cowan_update_kernel(
    float *E, float *I, const float *Conv_E, const float *Conv_I, int Nx,
    int Ny, float dt, float tau_E, float tau_I, float w_EE, float w_IE,
    float w_EI, float w_II, float P, float Q, float beta_E, float theta_E,
    float beta_I, float theta_I, float norm) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= Nx || y >= Ny)
    return;
  const int idx = y * Nx + x;

  const float cE = Conv_E[idx] * norm;
  const float cI = Conv_I[idx] * norm;

  const float drive_E = w_EE * cE - w_IE * cI + P;
  const float drive_I = w_EI * cE - w_II * cI + Q;

  const float sE = sigmoid(drive_E, beta_E, theta_E);
  const float sI = sigmoid(drive_I, beta_I, theta_I);

  const float e = E[idx];
  const float i = I[idx];
  E[idx] = e + dt * (-e + sE) / tau_E;
  I[idx] = i + dt * (-i + sI) / tau_I;
}

// ---------------------------------------------------------------------------
// colormap: scalar field -> RGB bytes (viridis-like)
// ---------------------------------------------------------------------------

__device__ __forceinline__ void viridis(float t, unsigned char &r,
                                        unsigned char &g, unsigned char &b) {
  // clamp
  t = fminf(fmaxf(t, 0.0f), 1.0f);
  // cheap 3-stop approximation to viridis
  // purple (0.267, 0.005, 0.329) -> teal (0.128, 0.567, 0.551) -> yellow (0.993, 0.906, 0.144)
  float rf, gf, bf;
  if (t < 0.5f) {
    const float s = t * 2.0f;
    rf = 0.267f * (1.0f - s) + 0.128f * s;
    gf = 0.005f * (1.0f - s) + 0.567f * s;
    bf = 0.329f * (1.0f - s) + 0.551f * s;
  } else {
    const float s = (t - 0.5f) * 2.0f;
    rf = 0.128f * (1.0f - s) + 0.993f * s;
    gf = 0.567f * (1.0f - s) + 0.906f * s;
    bf = 0.551f * (1.0f - s) + 0.144f * s;
  }
  r = (unsigned char)(255.0f * rf);
  g = (unsigned char)(255.0f * gf);
  b = (unsigned char)(255.0f * bf);
}

__global__ void colormap_rgb_kernel(unsigned char *rgb, const float *field,
                                    int Nx, int Ny, float vmin, float vmax) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= Nx || y >= Ny)
    return;
  const int idx = y * Nx + x;
  const float t = (field[idx] - vmin) / (vmax - vmin);
  unsigned char r, g, b;
  viridis(t, r, g, b);
  // flip y so image reads top-down like a plot
  const int out = ((Ny - 1 - y) * Nx + x) * 3;
  rgb[out + 0] = r;
  rgb[out + 1] = g;
  rgb[out + 2] = b;
}
