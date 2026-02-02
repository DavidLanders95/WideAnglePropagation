/*********************************************************************************/
/*                                                                               */
/*  Time-Domain Schrödinger/Helmholtz-like Solver (Split Real/Imag)              */
/*  CUDA ACCELERATED VERSION                                                     */
/*                                                                               */
/*  UPDATED: Top/Bottom PML using complex coordinate stretching (paper-style).   */
/*           - X remains periodic (as in your current code).                     */
/*           - Y uses a complex-stretched Laplacian in PML regions.              */
/*           - Replaces CAP + clamped stencil with a proper PML operator.        */
/*           - Uses double-buffered single-step update (complex operator mixes   */
/*             real/imag, so the old 2-kernel split is not valid with PML).      */
/*                                                                               */
/*********************************************************************************/

#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* General geometrical parameters */
#define WINWIDTH   720
#define WINHEIGHT  1280
#define NX 360
#define NY 640

#define XMIN -1.0000f
#define XMAX  1.0000f
#define YMIN -2.0000f
#define YMAX  2.0000f

/* Color parameters */
#define COLOR_SCHEME 3

/* Physical parameters */
#define DT 0.000001f
#define DX ((XMAX-XMIN)/((float)NX-1.0f))
#define DY ((YMAX-YMIN)/((float)NY-1.0f))

/* Relativistic / Helmholtz Parameters */
#define SOL_LIGHT 137.035999f   /* Speed of light (approx au) */
#define MASS_PART 1.0f          /* Particle Mass */
#define E_KINETIC 300.0f        /* Beam Energy */
/* Total Energy E = mc^2 + T */
#define E_TOTAL (MASS_PART*SOL_LIGHT*SOL_LIGHT + E_KINETIC)

/* Derived Beam Parameters */
#define P_INF (sqrtf(E_TOTAL*E_TOTAL - MASS_PART*MASS_PART*SOL_LIGHT*SOL_LIGHT*SOL_LIGHT*SOL_LIGHT) / SOL_LIGHT)
#define MOMENTUM_X P_INF
#define E_SIM (0.5f * MOMENTUM_X * MOMENTUM_X)

/* Source Injection Parameters */
#define J_SOURCE 120
#define SRC_BAND 3
#define SRC_ALPHA 0.05f      /* Coupling strength 0..1 */
#define PI_F 3.14159265f

/* ----------------------------- PML PARAMETERS ------------------------------ */
/* PML thickness (in coordinate units, same unit as X/Y). */
#define PML_THICK   0.5000f
/* Complex stretching angle theta. pi/4 is common. */
#define PML_THETA   0.7850f
/* Strength scale of sigma profile (tune). Paper example ~80. */
#define PML_SIGMA0  80.0000f
/* Polynomial ramp power for sigma profile. */
#define PML_POWER   3.0f
/* -------------------------------------------------------------------------- */

/* CUDA Error Checking Macro */
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

/* Device Helper: Atomic Potential (read from global memory array) */
__device__ __forceinline__ float potential_cuda(const float *d_V, int i, int j)
{
    return d_V[j * NX + i];
}

/* --------------------------- Complex helpers ------------------------------ */
__device__ __forceinline__ float2 cadd(float2 a, float2 b) { return make_float2(a.x + b.x, a.y + b.y); }
__device__ __forceinline__ float2 csub(float2 a, float2 b) { return make_float2(a.x - b.x, a.y - b.y); }
__device__ __forceinline__ float2 cmul(float2 a, float2 b)
{
    return make_float2(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}
__device__ __forceinline__ float2 cscale(float2 a, float s) { return make_float2(a.x*s, a.y*s); }
/* ------------------------------------------------------------------------- */

/* σ(y): 0 in interior, ramps in top/bottom PML of thickness PML_THICK */
__device__ __forceinline__ float sigma_pml_y(int j)
{
    float y = YMIN + ((float)j) * (YMAX - YMIN) / ((float)NY - 1.0f);

    float rb = fmaxf((YMIN + PML_THICK) - y, 0.0f) / PML_THICK; /* bottom depth 0..1 */
    float rt = fmaxf(y - (YMAX - PML_THICK), 0.0f) / PML_THICK; /* top depth 0..1 */
    float r  = fmaxf(rb, rt);

    if (r <= 0.0f) return 0.0f;

    return PML_SIGMA0 * powf(r, PML_POWER);
}

/* a(y) = 1 / S_y(y), where S_y = 1 + exp(iθ) σ(y)  */
__device__ __forceinline__ float2 a_over_Sy(int j)
{
    float sig = sigma_pml_y(j);
    if (sig <= 0.0f) return make_float2(1.0f, 0.0f);

    float c = cosf(PML_THETA);
    float s = sinf(PML_THETA);

    /* S = 1 + σ(c + i s) */
    float Sr = 1.0f + sig * c;
    float Si =        sig * s;

    /* 1/S = (Sr - iSi) / (Sr^2 + Si^2) */
    float den = Sr*Sr + Si*Si;
    return make_float2(Sr/den, -Si/den);
}

/* KERNEL: Soft internal source injection (plane-wave temporal phase)
   NOTE: This is a *soft* source, not a hard overwrite, to minimize reflections. */
__global__ void inject_source(float *d_phi, float *d_psi, float time, float E_sim_val)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (j >= NY || i >= NX) return;

    if (j >= J_SOURCE && j < (J_SOURCE + SRC_BAND))
    {
        int idx = j * NX + i;

        /* Beam momentum in +y direction */
        float k_beam = sqrtf(2.0f * E_sim_val);

        float y_source_center = YMIN + ((float)J_SOURCE) * (YMAX - YMIN) / ((float)NY - 1.0f);
        float y_curr = YMIN + ((float)j) * (YMAX - YMIN) / ((float)NY - 1.0f);

        /* Phase = k*(y - y0) - E*t */
        float phase = k_beam * (y_curr - y_source_center) - E_sim_val * time;

        float phi_inc = cosf(phase);
        float psi_inc = sinf(phase);

        /* smooth window across band */
        float s = 0.0f;
        if (SRC_BAND > 1) {
            s = ((float)(j - J_SOURCE)) / ((float)(SRC_BAND - 1));
        }
        float win = 0.5f * (1.0f - cosf(PI_F * s));
        float a = SRC_ALPHA * win;

        d_phi[idx] = (1.0f - a) * d_phi[idx] + a * phi_inc;
        d_psi[idx] = (1.0f - a) * d_psi[idx] + a * psi_inc;
    }
}

/* RK4 Single Stage Update
   Calculates k = H(phi_in, psi_in)
   accum += weight_accum * k   (Used for final integration)
   out   = base + step_dt * k (Used for next predictor step)
*/
__global__ void rk4_stage(
    const float *phi_base, const float *psi_base,   /* Base state for integration (U_n) */
    const float *phi_in,   const float *psi_in,     /* Input state for derivative (U_current) */
    float *phi_out, float *psi_out,                 /* Output state (U_next_predictor) - can be NULL */
    float *phi_accum, float *psi_accum,             /* Accumulator buffer */
    const float *d_V, 
    float step_dt,       /* dt factor for output update (e.g. 0.5*DT) */
    float weight_accum,  /* weight for accumulator (e.g. 2.0) */
    bool update_out      /* whether to write to phi_out/psi_out */
    )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= NX || j >= NY) return;

    int idx = j * NX + i;

    /* Boundary Conditions: Dirichlet at Top/Bottom */
    if (j == 0 || j == NY - 1) {
        if (update_out) {
            phi_out[idx] = 0.0f;
            psi_out[idx] = 0.0f;
        }
        /* Accumulator derivative is zero at boundary */
        /* But we must not leave uninitialized values if this is first stage? 
           Actually accum buffer is separate. If we add 0, it's fine. 
           Caller must clear accum at start of step. */
        return;
    }

    /* X periodic */
    int ip = (i + 1 == NX) ? 0 : (i + 1);
    int im = (i == 0) ? (NX - 1) : (i - 1);

    int jp = j + 1;
    int jm = j - 1;

    /* Read derivative source state */
    float ph = phi_in[idx];
    float ps = psi_in[idx];

    /* x-second-derivative (standard Laplacian part) */
    float d2x_ph = (phi_in[j*NX + ip] - 2.0f*ph + phi_in[j*NX + im]) / (DX*DX);
    float d2x_ps = (psi_in[j*NX + ip] - 2.0f*ps + psi_in[j*NX + im]) / (DX*DX);

    /* y operator: a_j * d/dy( a * dψ/dy ), with a=1/Sy */
    float2 a_j  = a_over_Sy(j);
    float2 a_jp = a_over_Sy(jp);
    float2 a_jm = a_over_Sy(jm);

    /* midpoint coefficients (simple averages) */
    float2 a_p = cscale(cadd(a_j, a_jp), 0.5f);
    float2 a_m = cscale(cadd(a_jm, a_j), 0.5f);

    float2 psi_c   = make_float2(ph, ps);
    float2 psi_jpC = make_float2(phi_in[jp*NX + i], psi_in[jp*NX + i]);
    float2 psi_jmC = make_float2(phi_in[jm*NX + i], psi_in[jm*NX + i]);

    /* q_{j+1/2} = a_p * (ψ_{j+1} - ψ_j)/DY */
    float2 dyp = cscale(csub(psi_jpC, psi_c), 1.0f / DY);
    float2 dym = cscale(csub(psi_c,   psi_jmC), 1.0f / DY);
    float2 q_p = cmul(a_p, dyp);
    float2 q_m = cmul(a_m, dym);

    /* Ly ψ ≈ a_j * (q_p - q_m)/DY */
    float2 dq = cscale(csub(q_p, q_m), 1.0f / DY);
    float2 Ly = cmul(a_j, dq);

    /* total Laplacian-like */
    float Lap_r = d2x_ph + Ly.x;
    float Lap_i = d2x_ps + Ly.y;

    float V = potential_cuda(d_V, i, j);

    /* Hψ = -0.5*Lap(ψ) + V ψ */
    float H_r = -0.5f * Lap_r + V * ph;
    float H_i = -0.5f * Lap_i + V * ps;

    /* Derivative k = -i H ψ */
    /* k_phi = Im(H); k_psi = -Re(H) */
    float k_phi = H_i;
    float k_psi = -H_r;
    
    /* Accumulate weighted k */
    /* Note: if weight_accum is 0, we skip. */
    /* Use atomicAdd? No, strict thread ownership 1:1. Read/Add/Write is safe. */
    phi_accum[idx] += weight_accum * k_phi;
    psi_accum[idx] += weight_accum * k_psi;

    /* Update Output for next stage */
    if (update_out) {
        /* out = base + step_dt * k */
        /* Make sure step_dt acts on k directly (k has units of 1/Time) */
        phi_out[idx] = phi_base[idx] + step_dt * k_phi;
        psi_out[idx] = psi_base[idx] + step_dt * k_psi;
    }
}

/* Final RK4 Update: U_n+1 = U_n + (DT/6) * Accum */
__global__ void rk4_final(float *d_phi, float *d_psi, const float *d_phi_accum, const float *d_psi_accum, float total_dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = j * NX + i;
    if (i >= NX || j >= NY) return;

    if (j == 0 || j == NY - 1) return; // Keep bound 0

    float factor = total_dt / 6.0f;
    d_phi[idx] += factor * d_phi_accum[idx];
    d_psi[idx] += factor * d_psi_accum[idx];
}

/* Host Functions for IO (unchanged) */
void write_ppm(char *filename, int width, int height, unsigned char *data) {
    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        printf("Error opening %s for writing\n", filename);
        return;
    }
    fprintf(fp, "P6\n%d %d\n255\n", width, height);
    fwrite(data, 1, width * height * 3, fp);
    fclose(fp);
    printf("Saved %s\n", filename);
}

void hsl_to_rgb(float h, float s, float l, float *r, float *g, float *b) {
    float c = (1.0f - fabsf(2.0f*l - 1.0f)) * s;
    float x = c * (1.0f - fabsf(fmodf(h/60.0f, 2.0f) - 1.0f));
    float m = l - c/2.0f;

    if(0<=h && h<60){ *r=c; *g=x; *b=0; }
    else if(60<=h && h<120){ *r=x; *g=c; *b=0; }
    else if(120<=h && h<180){ *r=0; *g=c; *b=x; }
    else if(180<=h && h<240){ *r=0; *g=x; *b=c; }
    else if(240<=h && h<300){ *r=x; *g=0; *b=c; }
    else{ *r=c; *g=0; *b=x; }

    *r += m; *g += m; *b += m;
}

void color_scheme(float phi, float psi, float scale, int time, float *rgb) {
    float mod = sqrtf(phi*phi + psi*psi);
    float val = mod * scale;
    if (val > 1.0f) val = 1.0f;

    float angle = atan2f(psi, phi) * 180.0f / 3.14159f;
    if(angle < 0) angle += 360.0f;

    hsl_to_rgb(angle, 1.0f, 0.5f * val, &rgb[0], &rgb[1], &rgb[2]);
}

int main(int argc, char **argv)
{
    int size = NX * NY * sizeof(float);
    float *h_phi = (float*)malloc(size);
    float *h_psi = (float*)malloc(size);
    float *h_V   = (float*)malloc(size);

    /* Load Potential Map */
    FILE *fp_pot = fopen("potential.bin", "rb");
    if(!fp_pot) {
        printf("Error: potential.bin not found. Please provide input potential map.\n");
        return 1;
    }
    fseek(fp_pot, 8, SEEK_SET);
    size_t read_count = fread(h_V, sizeof(float), NX*NY, fp_pot);
    if(read_count != (size_t)(NX*NY)) {
         printf("Warning: Potential file size mismatch? Read %zu floats.\n", read_count);
    }
    fclose(fp_pot);

    int i, frame;

    /* Parse command-line overrides: N_FRAMES STEPS_PER_FRAME SNAPSHOT_COUNT E_KINETIC */
    int N_FRAMES = 200;
    int STEPS_PER_FRAME = 50;
    int SNAPSHOT_COUNT = 10;
    float param_E_kinetic = E_KINETIC;

    if (argc > 1) N_FRAMES = atoi(argv[1]);
    if (argc > 2) STEPS_PER_FRAME = atoi(argv[2]);
    if (argc > 3) SNAPSHOT_COUNT = atoi(argv[3]);
    if (argc > 4) param_E_kinetic = atof(argv[4]);

    /* Recalculate derived physics parameters based on input energy
       If input < 0, assume it is WAVELENGTH in Angstroms */
    float param_E_total, param_p_inf, param_E_sim;
    float term_mass = MASS_PART * SOL_LIGHT * SOL_LIGHT;

    if (param_E_kinetic < 0) {
        float lambda_angstrom = -param_E_kinetic;
        float bohr_radius_angstrom = 0.5291772109f;
        float lambda_au = lambda_angstrom / bohr_radius_angstrom;

        float pi_val = 3.14159265359f;
        param_p_inf = (2.0f * pi_val) / lambda_au;

        float pc = param_p_inf * SOL_LIGHT;
        param_E_total = sqrtf(pc*pc + term_mass*term_mass);

        param_E_kinetic = param_E_total - term_mass;
        param_E_sim = 0.5f * param_p_inf * param_p_inf;

        printf("Input: Wavelength = %.4f Angstroms\n", lambda_angstrom);
    } else {
        param_E_total = term_mass + param_E_kinetic;
        param_p_inf = sqrtf(param_E_total*param_E_total - term_mass*term_mass) / SOL_LIGHT;
        param_E_sim = 0.5f * param_p_inf * param_p_inf;
    }

    printf("Initializing CUDA Simulation (Top/Bottom PML, X periodic)...\n");
    printf("Run: N_FRAMES=%d, STEPS_PER_FRAME=%d, SNAPSHOT_COUNT=%d\n", N_FRAMES, STEPS_PER_FRAME, SNAPSHOT_COUNT);
    printf("Physics: E_KINETIC=%.2f (=> E_SIM=%.4f), P_INF=%.4f\n", param_E_kinetic, param_E_sim, param_p_inf);
    printf("PML: THICK=%.3f, SIGMA0=%.2f, THETA=%.3f rad, POWER=%.1f\n",
           (float)PML_THICK, (float)PML_SIGMA0, (float)PML_THETA, (float)PML_POWER);
    printf("SRC: J_SOURCE=%d, SRC_BAND=%d, SRC_ALPHA=%.4f\n", (int)J_SOURCE, (int)SRC_BAND, (float)SRC_ALPHA);

    /* Initial Condition - Start with Vacuum */
    for (i=0; i<NX*NY; i++) {
        h_phi[i] = 0.0f;
        h_psi[i] = 0.0f;
    }

    /* Device Memory */
    float *d_phi, *d_psi, *d_phi2, *d_psi2, *d_phi3, *d_psi3, *d_V;
    cudaCheckError( cudaMalloc((void**)&d_phi,  size) );
    cudaCheckError( cudaMalloc((void**)&d_psi,  size) );
    cudaCheckError( cudaMalloc((void**)&d_phi2, size) );
    cudaCheckError( cudaMalloc((void**)&d_psi2, size) );
    cudaCheckError( cudaMalloc((void**)&d_phi3, size) );
    cudaCheckError( cudaMalloc((void**)&d_psi3, size) );
    cudaCheckError( cudaMalloc((void**)&d_V,    size) );

    /* Copy to Device */
    cudaCheckError( cudaMemcpy(d_phi,  h_phi, size, cudaMemcpyHostToDevice) );
    cudaCheckError( cudaMemcpy(d_psi,  h_psi, size, cudaMemcpyHostToDevice) );
    // d_phi2/3 initialized by use
    cudaCheckError( cudaMemcpy(d_V,    h_V,   size, cudaMemcpyHostToDevice) );

    /* Config */
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((NX + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (NY + threadsPerBlock.y - 1) / threadsPerBlock.y);

    /* Prepare snapshots file */
    FILE *fp_snap = fopen("snapshots.bin", "wb");
    if (!fp_snap) {
        printf("Error: cannot open snapshots.bin for writing\n");
        return 1;
    }
    int nx = NX, ny = NY, M = SNAPSHOT_COUNT;
    float dt_f = DT;
    float E_sim_file = param_E_sim;
    fwrite(&nx, sizeof(int), 1, fp_snap);
    fwrite(&ny, sizeof(int), 1, fp_snap);
    fwrite(&M,  sizeof(int), 1, fp_snap);
    fwrite(&dt_f, sizeof(float), 1, fp_snap);
    fwrite(&E_sim_file, sizeof(float), 1, fp_snap);

    int save_start = N_FRAMES - M;
    if (save_start < 0) save_start = 0;
    int saved = 0;

    /* Main Time Loop */
    for (frame=0; frame < N_FRAMES; frame++) {
        for (int s=0; s<STEPS_PER_FRAME; s++) {
            float time = (frame * STEPS_PER_FRAME + s) * DT;

            /* RK4 Integration Steps */

            /* Clear Accumulator at start of step */
            cudaCheckError( cudaMemset(d_phi3, 0, size) );
            cudaCheckError( cudaMemset(d_psi3, 0, size) );

            /* 0. Inject Source (modifies d_phi, d_psi in place) */
            inject_source<<<numBlocks, threadsPerBlock>>>(d_phi, d_psi, time, param_E_sim);

            /* Step 1 (k1):
               Input: U_n (d_phi/psi)
               Accum: + k1
               Out: U_n + 0.5*dt*k1 -> d_phi2/psi2
            */
            rk4_stage<<<numBlocks, threadsPerBlock>>>(
                d_phi, d_psi,      /* Base U_n */
                d_phi, d_psi,      /* Input (U_n) */
                d_phi2, d_psi2,    /* Output (U_n + k1*dt/2) */
                d_phi3, d_psi3,    /* Accum */
                d_V, 0.5f * DT,    /* dt_step */
                1.0f,              /* weight_accum (1.0 for k1) */
                true               /* update_out */
            );

            /* Step 2 (k2):
               Input: d_phi2/psi2 (U_n + 0.5 k1)
               Accum: + 2.0 * k2
               Out: U_n + 0.5*DT*k2 -> d_phi2/psi2 (reuse buffer)
            */
            rk4_stage<<<numBlocks, threadsPerBlock>>>(
                d_phi, d_psi,      /* Base U_n */
                d_phi2, d_psi2,    /* Input (Previous Stage) */
                d_phi2, d_psi2,    /* Output (U_n + k2*dt/2) */
                d_phi3, d_psi3,    /* Accum */
                d_V, 0.5f * DT,    /* dt_step */
                2.0f,              /* weight_accum */
                true               /* update_out */
            );

            /* Step 3 (k3):
               Input: d_phi2/psi2 (U_n + 0.5 k2)
               Accum: + 2.0 * k3
               Out: U_n + DT*k3 -> d_phi2/psi2
            */
            rk4_stage<<<numBlocks, threadsPerBlock>>>(
                d_phi, d_psi,      /* Base U_n */
                d_phi2, d_psi2,    /* Input */
                d_phi2, d_psi2,    /* Output (U_n + k3*dt) */
                d_phi3, d_psi3,    /* Accum */
                d_V, DT,           /* dt_step (Full Step) */
                2.0f,              /* weight_accum */
                true               /* update_out */
            );

            /* Step 4 (k4):
               Input: d_phi2/psi2 (U_n + k3)
               Accum: + 1.0 * k4
               Out: None
            */
            rk4_stage<<<numBlocks, threadsPerBlock>>>(
                d_phi, d_psi,      /* Base U_n */
                d_phi2, d_psi2,    /* Input */
                d_phi2, d_psi2,    /* Output (Ignored) */
                d_phi3, d_psi3,    /* Accum */
                d_V, 0.0f,         /* dt_step (Unused) */
                1.0f,              /* weight_accum */
                false              /* update_out */
            );
            
            /* Final Update: U_n += (DT/6)*Accum */
            rk4_final<<<numBlocks, threadsPerBlock>>>(
                d_phi, d_psi, 
                d_phi3, d_psi3, 
                DT
            );
        }

        if (frame % 20 == 0) printf("Progress: %d / %d frames\n", frame, N_FRAMES);

        /* Save snapshot if in the last M frames */
        if (frame >= save_start && saved < M) {
            cudaCheckError( cudaMemcpy(h_phi, d_phi, size, cudaMemcpyDeviceToHost) );
            cudaCheckError( cudaMemcpy(h_psi, d_psi, size, cudaMemcpyDeviceToHost) );
            float t_snapshot = (frame * STEPS_PER_FRAME) * DT;

            size_t w1 = fwrite(&t_snapshot, sizeof(float), 1, fp_snap);
            size_t w2 = fwrite(h_phi, sizeof(float), NX*NY, fp_snap);
            size_t w3 = fwrite(h_psi, sizeof(float), NX*NY, fp_snap);

            if (w1 != 1 || w2 != (size_t)(NX*NY) || w3 != (size_t)(NX*NY)) {
                 fprintf(stderr, "Error: Disk write failed at frame %d. Disk full?\n", frame);
                 exit(1); 
            }
            saved++;
        }
    }

    fclose(fp_snap);
    printf("Saved %d snapshots to snapshots.bin\n", saved);

    cudaFree(d_phi);
    cudaFree(d_psi);
    cudaFree(d_phi2);
    cudaFree(d_psi2);
    cudaFree(d_phi3);
    cudaFree(d_psi3);
    cudaFree(d_V);
    free(h_phi);
    free(h_psi);
    free(h_V);

    return 0;
}
