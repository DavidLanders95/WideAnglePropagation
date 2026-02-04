/*********************************************************************************/
/*                                                                               */
/*  3D Time-Domain Schrödinger/Helmholtz-like Solver                             */
/*  CUDA ACCELERATED VERSION                                                     */
/*                                                                               */
/*  Dimensions: X, Y (Transverse, Periodic)                                      */
/*              Z    (Propagation, PML + Dirichlet)                              */
/*                                                                               */
/*********************************************************************************/

#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* General geometrical parameters (Defaults, overwritten by Macros) */
#define NX 64
#define NY 64
#define NZ 203

#define XMIN -1.0000f
#define XMAX  1.0000f
#define YMIN -1.0000f
#define YMAX  1.0000f
#define ZMIN -1.6000f
#define ZMAX  1.5811f

/* Physical parameters */
#define DT 0.0000005f
#define DX ((XMAX-XMIN)/((float)NX-1.0f))
#define DY ((YMAX-YMIN)/((float)NY-1.0f))
#define DZ ((ZMAX-ZMIN)/((float)NZ-1.0f))

/* Relativistic / Helmholtz Parameters */
#define SOL_LIGHT 137.035999f
#define MASS_PART 1.0f
#define E_KINETIC 300.0f
/* Total Energy E = mc^2 + T */
#define E_TOTAL (MASS_PART*SOL_LIGHT*SOL_LIGHT + E_KINETIC)

/* Derived Beam Parameters */
#ifndef LAP_COEFF
#define LAP_COEFF 3.427719f
#endif

#ifndef P_INF
#define P_INF 125.663706f
#endif

// If LAP_COEFF / P_INF are defined externally (e.g. for eV/Angstrom), we use them.

/* Momentum along propagation axis Z */
#define MOMENTUM_Z P_INF
#define E_SIM (LAP_COEFF * MOMENTUM_Z * MOMENTUM_Z)

/* Source Injection Parameters (along Z) */
#define K_SOURCE 31
#define SRC_BAND 3
#define SRC_ALPHA 0.05f 
#define PI_F 3.14159265f

/* ----------------------------- PML PARAMETERS (Z-AXIS) -------------------- */
#define PML_THICK   0.4000f
#define PML_THETA   0.7850f
#define PML_SIGMA0  80.0000f
#define PML_POWER   3.0f
/* -------------------------------------------------------------------------- */

/* CUDA Error Checking Macro */
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

/* Helper: flattened index */
__device__ __forceinline__ int get_idx(int i, int j, int k) {
    return k * (NX * NY) + j * NX + i;
}

__device__ __forceinline__ float potential_cuda(const float *d_V, int idx) {
    return d_V[idx];
}

/* --------------------------- Complex helpers ------------------------------ */
__device__ __forceinline__ float2 cadd(float2 a, float2 b) { return make_float2(a.x + b.x, a.y + b.y); }
__device__ __forceinline__ float2 csub(float2 a, float2 b) { return make_float2(a.x - b.x, a.y - b.y); }
__device__ __forceinline__ float2 cmul(float2 a, float2 b) { return make_float2(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x); }
__device__ __forceinline__ float2 cscale(float2 a, float s) { return make_float2(a.x*s, a.y*s); }

/* σ(z): 0 in interior, ramps in top/bottom PML of thickness PML_THICK */
__device__ __forceinline__ float sigma_pml_z(int k)
{
    float z = ZMIN + ((float)k) * (ZMAX - ZMIN) / ((float)NZ - 1.0f);

    float rb = fmaxf((ZMIN + PML_THICK) - z, 0.0f) / PML_THICK; /* bottom (low z) */
    float rt = fmaxf(z - (ZMAX - PML_THICK), 0.0f) / PML_THICK; /* top (high z) */
    float r  = fmaxf(rb, rt);

    if (r <= 0.0f) return 0.0f;
    return PML_SIGMA0 * powf(r, PML_POWER);
}

/* a(z) = 1 / S_z(z) */
__device__ __forceinline__ float2 a_over_Sz(int k)
{
    float sig = sigma_pml_z(k);
    if (sig <= 0.0f) return make_float2(1.0f, 0.0f);

    float c = cosf(PML_THETA);
    float s = sinf(PML_THETA);

    /* S = 1 + σ(c + i s) */
    float Sr = 1.0f + sig * c;
    float Si =        sig * s;

    float den = Sr*Sr + Si*Si;
    return make_float2(Sr/den, -Si/den);
}

/* KERNEL: Source Injection (Plane wave along Z) */
__global__ void inject_source_3d(float *d_phi, float *d_psi, float time, float E_sim_val)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= NX || j >= NY || k >= NZ) return;

    if (k >= K_SOURCE && k < (K_SOURCE + SRC_BAND))
    {
        int idx = get_idx(i, j, k);

        /* MOMENTUM_Z is calculated by macro, but here we use sqrt(E_sim/Coeff) */
        float k_beam = sqrtf(E_sim_val / LAP_COEFF);

        float z_source_center = ZMIN + ((float)K_SOURCE) * (ZMAX - ZMIN) / ((float)NZ - 1.0f);
        float z_curr = ZMIN + ((float)k) * (ZMAX - ZMIN) / ((float)NZ - 1.0f);

        /* Phase = k*(z - z0) - E*t */
        float phase = k_beam * (z_curr - z_source_center) - E_sim_val * time;

        float phi_inc = cosf(phase);
        float psi_inc = sinf(phase);

        /* smooth window */
        float s = 0.0f;
        if (SRC_BAND > 1) {
            s = ((float)(k - K_SOURCE)) / ((float)(SRC_BAND - 1));
        }
        float win = 0.5f * (1.0f - cosf(PI_F * s));
        float a = SRC_ALPHA * win;

        d_phi[idx] = (1.0f - a) * d_phi[idx] + a * phi_inc;
        d_psi[idx] = (1.0f - a) * d_psi[idx] + a * psi_inc;
    }
}

/* RK4 Stage */
__global__ void rk4_stage_3d(
    const float *phi_base, const float *psi_base,
    const float *phi_in,   const float *psi_in,
    float *phi_out, float *psi_out,
    float *phi_accum, float *psi_accum,
    const float *d_V, 
    float step_dt,
    float weight_accum,
    bool update_out
    )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= NX || j >= NY || k >= NZ) return;

    int idx = get_idx(i, j, k);

    /* Z Boundaries: skip update to avoid out-of-bounds (PML handled in interior) */
    if (k == 0 || k == NZ - 1) {
        if (update_out) {
            phi_out[idx] = phi_base[idx];
            psi_out[idx] = psi_base[idx];
        }
        return;
    }

    /* Periodic X */
    int ip = (i + 1 == NX) ? 0 : (i + 1);
    int im = (i == 0) ? (NX - 1) : (i - 1);
    
    /* Periodic Y */
    int jp = (j + 1 == NY) ? 0 : (j + 1);
    int jm = (j == 0) ? (NY - 1) : (j - 1);

    int kp = k + 1;
    int km = k - 1;

    float ph = phi_in[idx];
    float ps = psi_in[idx];

    /* Laplacian X */
    float d2x_ph = (phi_in[get_idx(ip,j,k)] - 2.0f*ph + phi_in[get_idx(im,j,k)]) / (DX*DX);
    float d2x_ps = (psi_in[get_idx(ip,j,k)] - 2.0f*ps + psi_in[get_idx(im,j,k)]) / (DX*DX);

    /* Laplacian Y */
    float d2y_ph = (phi_in[get_idx(i,jp,k)] - 2.0f*ph + phi_in[get_idx(i,jm,k)]) / (DY*DY);
    float d2y_ps = (psi_in[get_idx(i,jp,k)] - 2.0f*ps + psi_in[get_idx(i,jm,k)]) / (DY*DY);

    /* Z Operator (PML) */
    float2 a_k  = a_over_Sz(k);
    float2 a_kp = a_over_Sz(kp);
    float2 a_km = a_over_Sz(km);

    float2 a_p = cscale(cadd(a_k, a_kp), 0.5f);
    float2 a_m = cscale(cadd(a_km, a_k), 0.5f);

    float2 psi_c   = make_float2(ph, ps);
    float2 psi_kpC = make_float2(phi_in[get_idx(i,j,kp)], psi_in[get_idx(i,j,kp)]);
    float2 psi_kmC = make_float2(phi_in[get_idx(i,j,km)], psi_in[get_idx(i,j,km)]);

    /* q_{k+1/2} = a_p * (ψ_{k+1} - ψ_k)/DZ */
    float2 dzp = cscale(csub(psi_kpC, psi_c), 1.0f / DZ);
    float2 dzm = cscale(csub(psi_c,   psi_kmC), 1.0f / DZ);
    float2 q_p = cmul(a_p, dzp);
    float2 q_m = cmul(a_m, dzm);

    /* Lz ψ ≈ a_k * (q_p - q_m)/DZ */
    float2 dq = cscale(csub(q_p, q_m), 1.0f / DZ);
    float2 Lz = cmul(a_k, dq);

    /* Total Laplacian */
    float Lap_r = d2x_ph + d2y_ph + Lz.x;
    float Lap_i = d2x_ps + d2y_ps + Lz.y;

    float V = potential_cuda(d_V, idx);

    /* Hψ = -LAP_COEFF * Lap(ψ) + V ψ */
    float H_r = -LAP_COEFF * Lap_r + V * ph;
    float H_i = -LAP_COEFF * Lap_i + V * ps;

    /* k = -i H ψ */
    float k_phi = H_i;
    float k_psi = -H_r;
    
    phi_accum[idx] += weight_accum * k_phi;
    psi_accum[idx] += weight_accum * k_psi;

    if (update_out) {
        phi_out[idx] = phi_base[idx] + step_dt * k_phi;
        psi_out[idx] = psi_base[idx] + step_dt * k_psi;
    }
}

__global__ void rk4_final_3d(float *d_phi, float *d_psi, const float *d_phi_accum, const float *d_psi_accum, float total_dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= NX || j >= NY || k >= NZ) return;
    int idx = get_idx(i, j, k);

    if (k == 0 || k == NZ - 1) return; 

    float factor = total_dt / 6.0f;
    d_phi[idx] += factor * d_phi_accum[idx];
    d_psi[idx] += factor * d_psi_accum[idx];
}

int main(int argc, char **argv)
{
    size_t num_elements = (size_t)NX * (size_t)NY * (size_t)NZ;
    size_t size_bytes = num_elements * sizeof(float);
    
    float *h_phi = (float*)malloc(size_bytes);
    float *h_psi = (float*)malloc(size_bytes);
    float *h_V   = (float*)malloc(size_bytes);

    /* Load Potential Map (Expect NX*NY*NZ floats) */
    FILE *fp_pot = fopen("potential.bin", "rb");
    if(!fp_pot) {
        printf("Error: potential.bin not found.\n");
        return 1;
    }
    /* Skip header (nx, ny, nz) */
    fseek(fp_pot, 3 * sizeof(int), SEEK_SET);
    size_t read_count = fread(h_V, sizeof(float), num_elements, fp_pot);
    if(read_count != num_elements) {
         printf("Warning: Potential file size mismatch? Read %zu floats, expected %zu.\n", read_count, num_elements);
    }
    fclose(fp_pot);

    int N_FRAMES = 200;
    int STEPS_PER_FRAME = 50;
    int SNAPSHOT_COUNT = 10;
    float param_E_kinetic = E_KINETIC;

    if (argc > 1) N_FRAMES = atoi(argv[1]);
    if (argc > 2) STEPS_PER_FRAME = atoi(argv[2]);
    if (argc > 3) SNAPSHOT_COUNT = atoi(argv[3]);
    if (argc > 4) param_E_kinetic = atof(argv[4]);

    /* Recalculate derived physics */
    // If we rely on passed P_INF or explicit Params via macro, 
    // we should be careful. 
    // Here we assume MACROS set P_INF correctly if we don't supply args.
    // Ideally we trust the Python wrapper to set Define Constants.

    float param_E_sim_calc;

    param_E_sim_calc = LAP_COEFF * P_INF * P_INF;

    printf("Initializing 3D Simulation (X, Y Periodic; Z PML)...\n");
    printf("Grid: %d x %d x %d\n", NX, NY, NZ);
    printf("E_SIM: %.4f (LapCoeff=%.4f)\n", param_E_sim_calc, LAP_COEFF);

    /* Init */
    for (size_t i=0; i<num_elements; i++) {
        h_phi[i] = 0.0f;
        h_psi[i] = 0.0f;
    }

    float *d_phi, *d_psi, *d_phi2, *d_psi2, *d_phi3, *d_psi3, *d_V;
    cudaCheckError( cudaMalloc((void**)&d_phi,  size_bytes) );
    cudaCheckError( cudaMalloc((void**)&d_psi,  size_bytes) );
    cudaCheckError( cudaMalloc((void**)&d_phi2, size_bytes) );
    cudaCheckError( cudaMalloc((void**)&d_psi2, size_bytes) );
    cudaCheckError( cudaMalloc((void**)&d_phi3, size_bytes) );
    cudaCheckError( cudaMalloc((void**)&d_psi3, size_bytes) );
    cudaCheckError( cudaMalloc((void**)&d_V,    size_bytes) );

    cudaCheckError( cudaMemcpy(d_phi,  h_phi, size_bytes, cudaMemcpyHostToDevice) );
    cudaCheckError( cudaMemcpy(d_psi,  h_psi, size_bytes, cudaMemcpyHostToDevice) );
    cudaCheckError( cudaMemcpy(d_V,    h_V,   size_bytes, cudaMemcpyHostToDevice) );

    dim3 threadsPerBlock(8, 8, 4);
    dim3 numBlocks((NX + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (NY + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (NZ + threadsPerBlock.z - 1) / threadsPerBlock.z);

    FILE *fp_snap = fopen("snapshots.bin", "wb");
    if (!fp_snap) { printf("Error opening snapshots.bin\n"); return 1; }
    
    int nx = NX, ny = NY, nz = NZ, M = SNAPSHOT_COUNT;
    float dt_f = DT;
    float E_sim_file = param_E_sim_calc;
    fwrite(&nx, sizeof(int), 1, fp_snap);
    fwrite(&ny, sizeof(int), 1, fp_snap);
    fwrite(&nz, sizeof(int), 1, fp_snap);
    fwrite(&M,  sizeof(int), 1, fp_snap);
    fwrite(&dt_f, sizeof(float), 1, fp_snap);
    fwrite(&E_sim_file, sizeof(float), 1, fp_snap);

    int save_start = N_FRAMES - M;
    if (save_start < 0) save_start = 0;
    int saved = 0;

    for (int frame=0; frame < N_FRAMES; frame++) {
        for (int s=0; s<STEPS_PER_FRAME; s++) {
            float time = (frame * STEPS_PER_FRAME + s) * DT;

            cudaCheckError( cudaMemset(d_phi3, 0, size_bytes) );
            cudaCheckError( cudaMemset(d_psi3, 0, size_bytes) );

            inject_source_3d<<<numBlocks, threadsPerBlock>>>(d_phi, d_psi, time, param_E_sim_calc);

            rk4_stage_3d<<<numBlocks, threadsPerBlock>>>(
                d_phi, d_psi, d_phi, d_psi, d_phi2, d_psi2, d_phi3, d_psi3, d_V, 0.5f*DT, 1.0f, true);
            rk4_stage_3d<<<numBlocks, threadsPerBlock>>>(
                d_phi, d_psi, d_phi2, d_psi2, d_phi2, d_psi2, d_phi3, d_psi3, d_V, 0.5f*DT, 2.0f, true);
            rk4_stage_3d<<<numBlocks, threadsPerBlock>>>(
                d_phi, d_psi, d_phi2, d_psi2, d_phi2, d_psi2, d_phi3, d_psi3, d_V, DT, 2.0f, true);
            rk4_stage_3d<<<numBlocks, threadsPerBlock>>>(
                d_phi, d_psi, d_phi2, d_psi2, d_phi2, d_psi2, d_phi3, d_psi3, d_V, 0.0f, 1.0f, false);
            
            rk4_final_3d<<<numBlocks, threadsPerBlock>>>(d_phi, d_psi, d_phi3, d_psi3, DT);
        }

        if (frame % 10 == 0) printf("Frame %d / %d\n", frame, N_FRAMES);

        if (frame >= save_start && saved < M) {
            cudaCheckError( cudaMemcpy(h_phi, d_phi, size_bytes, cudaMemcpyDeviceToHost) );
            cudaCheckError( cudaMemcpy(h_psi, d_psi, size_bytes, cudaMemcpyDeviceToHost) );
            float t_snapshot = (frame * STEPS_PER_FRAME) * DT;
            fwrite(&t_snapshot, sizeof(float), 1, fp_snap);
            fwrite(h_phi, sizeof(float), num_elements, fp_snap);
            fwrite(h_psi, sizeof(float), num_elements, fp_snap);
            saved++;
        }
    }

    fclose(fp_snap);
    printf("Done. Saved %d snapshots.\n", saved);
    
    cudaFree(d_phi); cudaFree(d_psi); cudaFree(d_V);
    free(h_phi); free(h_psi); free(h_V);
    return 0;
}
