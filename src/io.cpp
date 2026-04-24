/*
 * io.cpp - host-side frame writing, preset table, argument parsing
 */

#include "io.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>

// ---------------------------------------------------------------------------
// PPM P6 writer
// ---------------------------------------------------------------------------

void write_ppm(const std::string &path, const unsigned char *rgb, int W,
               int H) {
  std::ofstream f(path, std::ios::binary);
  if (!f) {
    std::fprintf(stderr, "write_ppm: cannot open %s\n", path.c_str());
    return;
  }
  f << "P6\n" << W << " " << H << "\n255\n";
  f.write(reinterpret_cast<const char *>(rgb), (size_t)W * H * 3);
}

// ---------------------------------------------------------------------------
// presets -- hand-picked parameter sets that land in different regimes of
// the Wilson-Cowan phase diagram.  These are starting points; tuning is
// expected as part of the project.
// ---------------------------------------------------------------------------

bool apply_preset(WCParams &p, const std::string &name) {
  p.preset_name = name;
  if (name == "spots") {
    p.sigma_E = 1.5f; p.sigma_I = 4.0f;
    p.w_EE = 16.0f;   p.w_IE = 12.0f;
    p.w_EI = 15.0f;   p.w_II = 3.0f;
    p.P = 1.25f;
    return true;
  }
  if (name == "stripes") {
    p.sigma_E = 1.8f; p.sigma_I = 4.5f;
    p.w_EE = 14.0f;   p.w_IE = 13.0f;
    p.w_EI = 14.0f;   p.w_II = 4.0f;
    p.P = 1.1f;
    return true;
  }
  if (name == "spirals") {
    p.sigma_E = 2.0f; p.sigma_I = 5.0f;
    p.w_EE = 18.0f;   p.w_IE = 14.0f;
    p.w_EI = 16.0f;   p.w_II = 2.5f;
    p.P = 1.35f;
    p.bump_amp = 0.4f; p.bump_sigma = 6.0f;
    return true;
  }
  if (name == "waves") {
    p.sigma_E = 2.5f; p.sigma_I = 6.0f;
    p.w_EE = 15.0f;   p.w_IE = 15.0f;
    p.w_EI = 13.0f;   p.w_II = 3.5f;
    p.P = 1.2f;
    p.tau_I = 2.0f;
    return true;
  }
  if (name == "default") {
    return true;  // keep the struct defaults
  }
  return false;
}

// ---------------------------------------------------------------------------
// command-line parsing
// ---------------------------------------------------------------------------

static bool match(const char *arg, const char *key, const char *&val) {
  const size_t n = std::strlen(key);
  if (std::strncmp(arg, key, n) != 0) return false;
  if (arg[n] != '=') return false;
  val = arg + n + 1;
  return true;
}

void parse_args(WCParams &p, int argc, char **argv) {
  for (int i = 1; i < argc; ++i) {
    const char *a = argv[i];
    const char *v = nullptr;

    if (match(a, "--preset", v))        { apply_preset(p, v); }
    else if (match(a, "--nx", v))       { p.Nx = std::atoi(v); }
    else if (match(a, "--ny", v))       { p.Ny = std::atoi(v); }
    else if (match(a, "--nsteps", v))   { p.nsteps = std::atoi(v); }
    else if (match(a, "--frame-stride", v)) { p.frame_stride = std::atoi(v); }
    else if (match(a, "--dt", v))       { p.dt = (float)std::atof(v); }
    else if (match(a, "--sigma-e", v))  { p.sigma_E = (float)std::atof(v); }
    else if (match(a, "--sigma-i", v))  { p.sigma_I = (float)std::atof(v); }
    else if (match(a, "--w-ee", v))     { p.w_EE = (float)std::atof(v); }
    else if (match(a, "--w-ie", v))     { p.w_IE = (float)std::atof(v); }
    else if (match(a, "--w-ei", v))     { p.w_EI = (float)std::atof(v); }
    else if (match(a, "--w-ii", v))     { p.w_II = (float)std::atof(v); }
    else if (match(a, "--p", v))        { p.P = (float)std::atof(v); }
    else if (match(a, "--q", v))        { p.Q = (float)std::atof(v); }
    else if (match(a, "--beta-e", v))   { p.beta_E = (float)std::atof(v); }
    else if (match(a, "--theta-e", v))  { p.theta_E = (float)std::atof(v); }
    else if (match(a, "--beta-i", v))   { p.beta_I = (float)std::atof(v); }
    else if (match(a, "--theta-i", v))  { p.theta_I = (float)std::atof(v); }
    else if (match(a, "--tau-e", v))    { p.tau_E = (float)std::atof(v); }
    else if (match(a, "--tau-i", v))    { p.tau_I = (float)std::atof(v); }
    else if (match(a, "--seed", v))     { p.seed = (unsigned int)std::strtoul(v, nullptr, 10); }
    else if (match(a, "--out", v))      { p.out_dir = v; }
    else {
      std::fprintf(stderr, "unknown arg: %s (ignored)\n", a);
    }
  }
}

void print_params(const WCParams &p) {
  std::printf("preset       = %s\n", p.preset_name.c_str());
  std::printf("grid         = %d x %d\n", p.Nx, p.Ny);
  std::printf("dt, nsteps   = %g, %d  (frame stride %d)\n", p.dt, p.nsteps, p.frame_stride);
  std::printf("tau_E, tau_I = %g, %g\n", p.tau_E, p.tau_I);
  std::printf("w_EE w_IE    = %g %g\n", p.w_EE, p.w_IE);
  std::printf("w_EI w_II    = %g %g\n", p.w_EI, p.w_II);
  std::printf("sigma_E/I    = %g / %g\n", p.sigma_E, p.sigma_I);
  std::printf("beta/theta E = %g / %g\n", p.beta_E, p.theta_E);
  std::printf("beta/theta I = %g / %g\n", p.beta_I, p.theta_I);
  std::printf("P, Q         = %g, %g\n", p.P, p.Q);
  std::printf("noise, bump  = %g, %g (sigma %g)\n", p.noise_amp, p.bump_amp, p.bump_sigma);
  std::printf("seed         = %u\n", p.seed);
  std::printf("out_dir      = %s\n", p.out_dir.c_str());
}
