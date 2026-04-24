/*
 * main.cu - Wilson-Cowan neural field simulation on GPU
 *
 * Solves the Wilson-Cowan equations for excitatory and inhibitory activity
 * fields on a 2D periodic grid:
 *
 *   tau_E dE/dt = -E + S( w_EE*(K_E * E) - w_IE*(K_I * I) + P,  beta_E, theta_E )
 *   tau_I dI/dt = -I + S( w_EI*(K_E * E) - w_II*(K_I * I) + Q,  beta_I, theta_I )
 *
 * Spatial coupling is a Gaussian convolution evaluated by FFT using cuFFT.
 * The Gaussian Fourier transforms are computed analytically in k-space
 * (no forward FFT of the kernel needed).  Time stepping is forward Euler.
 *
 * Each time step costs 2 R2C FFTs + 2 C2R FFTs + a handful of pointwise
 * kernels.  Output is written as binary PPM frames at a configurable stride;
 * stitch them into video with scripts/make_video.sh (ffmpeg).
 */

#include "io.hpp"
#include "kernels.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <cufft.h>
#include <chrono>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// error checking
// ---------------------------------------------------------------------------

#define CU_CHECK(code)                                                         \
  do {                                                                         \
    cudaError_t err = (code);                                                  \
    if (err != cudaSuccess) {                                                  \
      std::fprintf(stderr, "CUDA error <<%s>> at %s:%d\n",                     \
                   cudaGetErrorString(err), __FILE__, __LINE__);               \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

#define FFT_CHECK(code)                                                        \
  do {                                                                         \
    cufftResult res = (code);                                                  \
    if (res != CUFFT_SUCCESS) {                                                \
      std::fprintf(stderr, "cuFFT error %d at %s:%d\n", (int)res, __FILE__,    \
                   __LINE__);                                                  \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

// ---------------------------------------------------------------------------
// launch-config helpers
// ---------------------------------------------------------------------------

static inline dim3 grid2d(int Nx, int Ny, dim3 block) {
  return dim3((Nx + block.x - 1) / block.x, (Ny + block.y - 1) / block.y);
}

// ---------------------------------------------------------------------------
// driver
// ---------------------------------------------------------------------------

int main(int argc, char **argv) {
  using clk = std::chrono::steady_clock;

  WCParams p;
  parse_args(p, argc, argv);
  print_params(p);

  const int Nx = p.Nx;
  const int Ny = p.Ny;
  const int Nxh = Nx / 2 + 1;        // R2C compressed x-dim (inner/fastest)
  const size_t n_real = (size_t)Nx * Ny;
  const size_t n_cplx = (size_t)Ny * Nxh;

  // ---- allocate device buffers ---------------------------------------------
  float *d_E = nullptr, *d_I = nullptr;
  float *d_ConvE = nullptr, *d_ConvI = nullptr;
  cufftComplex *d_KE_hat = nullptr, *d_KI_hat = nullptr;
  cufftComplex *d_work = nullptr;     // reused buffer for FFT(E), FFT(I)
  unsigned char *d_rgb = nullptr;

  CU_CHECK(cudaMalloc(&d_E,     n_real * sizeof(float)));
  CU_CHECK(cudaMalloc(&d_I,     n_real * sizeof(float)));
  CU_CHECK(cudaMalloc(&d_ConvE, n_real * sizeof(float)));
  CU_CHECK(cudaMalloc(&d_ConvI, n_real * sizeof(float)));
  CU_CHECK(cudaMalloc(&d_KE_hat, n_cplx * sizeof(cufftComplex)));
  CU_CHECK(cudaMalloc(&d_KI_hat, n_cplx * sizeof(cufftComplex)));
  CU_CHECK(cudaMalloc(&d_work,   n_cplx * sizeof(cufftComplex)));
  CU_CHECK(cudaMalloc(&d_rgb,    3 * n_real));

  std::vector<unsigned char> h_rgb(3 * n_real);

  // ---- cuFFT plans ---------------------------------------------------------
  // 2D R2C expects dimensions (Ny, Nx) in row-major; we treat y as slowest.
  cufftHandle plan_r2c, plan_c2r;
  FFT_CHECK(cufftPlan2d(&plan_r2c, Ny, Nx, CUFFT_R2C));
  FFT_CHECK(cufftPlan2d(&plan_c2r, Ny, Nx, CUFFT_C2R));

  // ---- build the two Gaussian kernels in Fourier space ---------------------
  {
    dim3 block(16, 16);
    dim3 grid(grid2d(Nxh, Ny, block));
    gaussian_kernel_hat_kernel<<<grid, block>>>(d_KE_hat, Nx, Ny, p.sigma_E,
                                                p.dx, p.dy);
    gaussian_kernel_hat_kernel<<<grid, block>>>(d_KI_hat, Nx, Ny, p.sigma_I,
                                                p.dx, p.dy);
    CU_CHECK(cudaGetLastError());
  }

  // ---- initial conditions --------------------------------------------------
  {
    dim3 block(16, 16);
    dim3 grid(grid2d(Nx, Ny, block));
    init_fields_kernel<<<grid, block>>>(d_E, d_I, Nx, Ny, p.noise_amp,
                                        p.bump_amp, p.bump_sigma, p.seed);
    CU_CHECK(cudaGetLastError());
  }

  // ---- output directory ----------------------------------------------------
  // belt-and-suspenders: the shell scripts already mkdir the out dir, but
  // run the binary standalone and this keeps it self-sufficient.  Shell-out
  // avoids a C++17 <filesystem>/libstdc++fs link dependency on GCC 8.
  {
    std::string cmd = "mkdir -p \"" + p.out_dir + "\"";
    if (std::system(cmd.c_str()) != 0) {
      std::fprintf(stderr, "warning: mkdir -p %s failed\n", p.out_dir.c_str());
    }
  }

  // ---- time loop -----------------------------------------------------------
  const float norm = 1.0f / (float)(n_real);
  dim3 block2(16, 16);
  dim3 grid2(grid2d(Nx, Ny, block2));
  const int block1 = 256;
  const int grid1 = ((int)n_cplx + block1 - 1) / block1;

  auto t_start = clk::now();
  int frame_id = 0;

  for (int step = 0; step < p.nsteps; ++step) {

    // --- Conv_E = K_E * E  ---
    FFT_CHECK(cufftExecR2C(plan_r2c, d_E, d_work));
    multiply_hat_kernel<<<grid1, block1>>>(d_work, d_KE_hat, (int)n_cplx);
    FFT_CHECK(cufftExecC2R(plan_c2r, d_work, d_ConvE));

    // --- Conv_I = K_I * I  ---
    FFT_CHECK(cufftExecR2C(plan_r2c, d_I, d_work));
    multiply_hat_kernel<<<grid1, block1>>>(d_work, d_KI_hat, (int)n_cplx);
    FFT_CHECK(cufftExecC2R(plan_c2r, d_work, d_ConvI));

    // --- Wilson-Cowan Euler update ---
    wilson_cowan_update_kernel<<<grid2, block2>>>(
        d_E, d_I, d_ConvE, d_ConvI, Nx, Ny, p.dt, p.tau_E, p.tau_I, p.w_EE,
        p.w_IE, p.w_EI, p.w_II, p.P, p.Q, p.beta_E, p.theta_E, p.beta_I,
        p.theta_I, norm);

    // --- optionally write a frame ---
    if (step % p.frame_stride == 0) {
      colormap_rgb_kernel<<<grid2, block2>>>(d_rgb, d_E, Nx, Ny, -0.2f, 1.2f);
      CU_CHECK(cudaMemcpy(h_rgb.data(), d_rgb, 3 * n_real,
                          cudaMemcpyDeviceToHost));
      char name[512];
      std::snprintf(name, sizeof(name), "%s/frame_%05d.ppm",
                    p.out_dir.c_str(), frame_id);
      write_ppm(name, h_rgb.data(), Nx, Ny);
      ++frame_id;
    }
  }
  CU_CHECK(cudaDeviceSynchronize());
  auto t_end = clk::now();

  const double ms = std::chrono::duration<double, std::milli>(t_end - t_start)
                        .count();
  const double step_ms = ms / (double)p.nsteps;
  std::printf("\ntotal         : %.2f s  (%d steps, %d frames)\n", ms * 1e-3,
              p.nsteps, frame_id);
  std::printf("per step avg  : %.3f ms\n", step_ms);
  const double throughput_Gcell_s =
      (double)n_real / (step_ms * 1.0e6);  // cells per ms -> Gcells/s
  std::printf("throughput    : %.2f Gcells/s\n", throughput_Gcell_s);

  // machine-readable summary line for benchmark post-processing
  std::printf("CSV,%d,%d,%d,%.6f,%.6f,%.6f\n", Nx, Ny, p.nsteps, ms * 1e-3,
              step_ms, throughput_Gcell_s);

  // ---- cleanup -------------------------------------------------------------
  cufftDestroy(plan_r2c);
  cufftDestroy(plan_c2r);
  cudaFree(d_E);
  cudaFree(d_I);
  cudaFree(d_ConvE);
  cudaFree(d_ConvI);
  cudaFree(d_KE_hat);
  cudaFree(d_KI_hat);
  cudaFree(d_work);
  cudaFree(d_rgb);
  return 0;
}
