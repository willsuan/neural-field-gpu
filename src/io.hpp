/*
 * io.hpp - host-side utilities: PPM frame writing, config parsing
 */

#pragma once

#include <string>

// Write binary PPM (P6) from an RGB byte buffer of shape (H, W, 3).
void write_ppm(const std::string &path, const unsigned char *rgb, int W, int H);

// Parameters for one Wilson-Cowan run
struct WCParams {
  int Nx = 1024;
  int Ny = 1024;
  float dx = 1.0f;
  float dy = 1.0f;
  float dt = 0.1f;
  int nsteps = 4000;
  int frame_stride = 20;     // write one frame every N steps

  float tau_E = 1.0f;
  float tau_I = 1.0f;
  float w_EE = 16.0f;
  float w_IE = 12.0f;
  float w_EI = 15.0f;
  float w_II = 3.0f;
  float sigma_E = 1.5f;      // short-range excitation, grid units
  float sigma_I = 4.0f;      // longer-range inhibition
  float beta_E = 1.3f;
  float theta_E = 4.0f;
  float beta_I = 2.0f;
  float theta_I = 3.7f;
  float P = 1.25f;
  float Q = 0.0f;

  float noise_amp = 0.1f;
  float bump_amp = 0.2f;
  float bump_sigma = 8.0f;
  unsigned int seed = 12345u;

  std::string out_dir = "frames";
  std::string preset_name = "default";
};

// Apply a named preset to a WCParams in place.  Returns true if the name
// matched a known preset, false otherwise.
bool apply_preset(WCParams &p, const std::string &name);

// Parse --key=value style command-line arguments into an existing WCParams.
// Recognized keys mirror the struct fields.  Unknown keys print a warning.
void parse_args(WCParams &p, int argc, char **argv);

// Print the active parameters to stdout for reproducibility.
void print_params(const WCParams &p);
