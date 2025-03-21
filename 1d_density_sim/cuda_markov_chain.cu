#include <math.h>
#include <unistd.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <string.h>

#include <random>
#include <iostream>
#include <cuda.h>
#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/device_malloc.h>
#include <thrust/sort.h>
#include <thrust/binary_search.h>
#include <thrust/adjacent_difference.h>
#include <cuda_runtime.h>

#define ppi (3.14159265358979324)
#define MaxThreads (512)

__global__ void init_p(double *pp, double *pm, const int N);
__global__ void init_FV(double *F_list, double *VL, double *VC, double *VR, const int N, const double Lsize, const double dl, const double Dt, const double v);
__global__ void init_W_Cv(double *VL, double *VC, double *VR, double *Wlp, double *Wlm, double *Wrp, double *Wrm, const int N, const double dt, const double dl, const double Dt, const double v);
__global__ void init_W_C0pot(double *VL, double *VC, double *VR, double *Wlp, double *Wlm, double *Wrp, double *Wrm, const int N, const double dt, const double dl, const double Dt, const double v);
__global__ void update(double *pp, double *pm, double *Wlp, double *Wlm, double *Wrp, double *Wrm, double Wflip, const int N);
void save_current_state(double *p, const int N, char *filename);
__global__ void EP(double *pp, double *pm, double *VL, double *VC, double *VR, double *Wlp, double *Wlm, double *Wrp, double *Wrm, double *EPp, double *EPm, const int N, const double dt, const double dl, const double Dt, const double v);
void save_current_state(double *pp, double *pm, double *EPp, double *EPm, const int N, char *filename);

void error_output(const char *desc)
{
    printf("%s\n", desc) ; exit(-1) ;
}

int main(int argc, char *argv[])
{
    srand(time(NULL));
    // device setting
    const int    device_num = atoi(argv[1]);
    cudaSetDevice(device_num);
        
    // folder setting
    char folder_name[100];
    sprintf(folder_name, "%s", argv[2]);
    mkdir(folder_name, S_IRWXU);

     
    // Lattice and particle setting
    const double  L_size  = atoi(argv[3]);  // real_length
    const int       Tmin  = atoi(argv[4]);  // real starting time
    const int       Tmax  = atoi(argv[5]);  // real ending time

    // Dynamics setting
    const int     n      = atoi(argv[6]);   
    const double  dt     = 0.1/pow(4,n);
    const double  dl     = 0.25/pow(2,n);
    const double  speed  = 1.0;

    const double  alpha  = 0.02; // tumbling rate
    const double  Dt     = atof(argv[7]); // translational diffusion coefficient
    double Wflip         = 0.5 * (1.0 - exp(-alpha * dt));
   
    const int     type   = atoi(argv[8]); // Cv or C0pot
    if(type < 0 || type > 1) error_output("invalid type") ;
    
    const long freq = (long)(1/dt);
    // const long tmin = (long)(Tmin/dt);
    // const long tmax = (long)(Tmax/dt);    // real_time/dt


    // grid dimension
    int N = (int)(L_size/dl);

    const int nThreads = (MaxThreads<N)? MaxThreads : N;
    const int nBlocks  = (N+nThreads-1)/nThreads; 
    
    double *host_pm;
    double *host_pp;
    host_pm = (double *)malloc(sizeof(double) * N);
    host_pp = (double *)malloc(sizeof(double) * N);

    double *dev_pm;
    double *dev_pp;
    cudaMalloc(&dev_pm, sizeof(double) * N);
    cudaMalloc(&dev_pp, sizeof(double) * N);
    
    double *F; double *VL; double *VC; double *VR;
    cudaMalloc(&F, sizeof(double) * N);
    cudaMalloc(&VL, sizeof(double) * N);
    cudaMalloc(&VC, sizeof(double) * N);
    cudaMalloc(&VR, sizeof(double) * N);
    
    double *host_F; double *host_VL; double *host_VC; double *host_VR;
    host_F = (double *)malloc(sizeof(double) * N);
    host_VL = (double *)malloc(sizeof(double) * N);
    host_VC = (double *)malloc(sizeof(double) * N);
    host_VR = (double *)malloc(sizeof(double) * N);
    
    double *Wlp; double *Wlm; double *Wrp; double *Wrm;
    cudaMalloc(&Wlp, sizeof(double) * N);
    cudaMalloc(&Wlm, sizeof(double) * N);
    cudaMalloc(&Wrp, sizeof(double) * N);
    cudaMalloc(&Wrm, sizeof(double) * N);
    
    double *host_Wlp; double *host_Wlm; double *host_Wrp; double *host_Wrm;
    host_Wlp = (double *)malloc(sizeof(double) * N);
    host_Wlm = (double *)malloc(sizeof(double) * N);
    host_Wrp = (double *)malloc(sizeof(double) * N);
    host_Wrm = (double *)malloc(sizeof(double) * N);
    
    double *EPp; double *EPm;
    cudaMalloc(&EPp, sizeof(double) * N);
    cudaMalloc(&EPm, sizeof(double) * N);
    
    double *host_EPp; double *host_EPm;
    host_EPp = (double *)malloc(sizeof(double) * N);
    host_EPm = (double *)malloc(sizeof(double) * N);
    
    init_p<<<nBlocks, nThreads>>>(dev_pp, dev_pm, N);
    init_FV<<<nBlocks, nThreads>>>(F, VL, VC, VR, N, L_size, dl, Dt, speed);
    if (type == 0) {init_W_Cv<<<nBlocks, nThreads>>>(VL, VC, VR, Wlp, Wlm, Wrp, Wrm, N, dt, dl, Dt, speed);}
    if (type == 1) {init_W_C0pot<<<nBlocks, nThreads>>>(VL, VC, VR, Wlp, Wlm, Wrp, Wrm, N, dt, dl, Dt, speed);}
    
    cudaMemcpy(host_F, F, sizeof(double) * N, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_VL, VL, sizeof(double) * N, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_VC, VC, sizeof(double) * N, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_VR, VR, sizeof(double) * N, cudaMemcpyDeviceToHost);
    
    
    char Fname[200];
    sprintf(Fname, "%s/FV.txt", folder_name);
    FILE *fp = fopen(Fname, "w");
    for (int i = 0; i < N; i++){
        fprintf(fp, "%.10f %.10f %.10f %.10f ", host_F[i], host_VL[i], host_VC[i], host_VR[i]);
    }
    fclose(fp);

    cudaMemcpy(host_Wlp, Wlp, sizeof(double) * N, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_Wrp, Wrp, sizeof(double) * N, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_Wlm, Wlm, sizeof(double) * N, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_Wrm, Wrm, sizeof(double) * N, cudaMemcpyDeviceToHost);

    char Wname[200];
    sprintf(Wname, "%s/W.txt", folder_name);
    fp = fopen(Wname, "w");
    for (int i = 0; i < N; i++){
        fprintf(fp, "%.10f %.10f %.10f %.10f ", host_Wlp[i], host_Wlm[i], host_Wrp[i], host_Wrm[i]);
    }
    fclose(fp);

    
    for (int t = Tmin; t <= Tmax; t++){
        for (long tt = 0; tt < freq; tt++){
            update<<<nBlocks, nThreads>>>(dev_pp, dev_pm, Wlp, Wlm, Wrp, Wrm, Wflip, N);
        }
        
        if ((t%1000)==0 && t>= 0) {
            cudaMemcpy(host_pp, dev_pp, sizeof(double) * N, cudaMemcpyDeviceToHost);
            cudaMemcpy(host_pm, dev_pm, sizeof(double) * N, cudaMemcpyDeviceToHost);
            
            EP<<<nBlocks, nThreads>>>(dev_pp, dev_pm, VL, VC, VR, Wlp, Wlm, Wrp, Wrm, EPp, EPm, N, dt, dl, Dt, speed);
            
            cudaMemcpy(host_EPp, EPp, sizeof(double) * N, cudaMemcpyDeviceToHost);
            cudaMemcpy(host_EPm, EPm, sizeof(double) * N, cudaMemcpyDeviceToHost);

            char state_filename[200];
            sprintf(state_filename, "%s/state_t_%d.txt", folder_name, t);
            save_current_state(host_pp, host_pm, host_EPp, host_EPm, N, state_filename);
            printf("t = %d\n", t);
            
        }
    }

    cudaFree(dev_pp); cudaFree(dev_pm); free(host_pp); free(host_pm); cudaFree(F); cudaFree(VL); cudaFree(VC); cudaFree(VR); free(host_F); free(host_VL); free(host_VC); free(host_VR); cudaFree(Wlp); cudaFree(Wrp); cudaFree(Wlm); cudaFree(Wrm); cudaFree(EPp); cudaFree(EPm); free(host_EPp); free(host_EPm); free(host_Wlp); free(host_Wrp); free(host_Wlm); free(host_Wrm);
    

}

__global__ void init_p(double *pp, double *pm, const int N){
    const int tid = threadIdx.x + blockDim.x * blockIdx.x ;
    if(tid < N) {
        pp[tid] = (double) 0.5/N;
        pm[tid] = (double) 0.5/N;
    }
}

__global__ void init_FV(double *F, double *VL, double *VC, double *VR, const int N, const double Lsize, const double dl, const double Dt, const double v){
    const int tid = threadIdx.x + blockDim.x * blockIdx.x ;

    double tempF, V, V_left, V_right; // V is potential normalized by temperature.

    // size of the leftside and rightside of the potential
    double x_real, x_left, x_right; //actual position not grid coordinate
    double mid_pt = Lsize*0.5;
    double X, Xl, Xr;

    if(tid < N) {
        
        x_real = (double)(tid)*dl;
        x_left = x_real - dl;
        x_right = x_real + dl;
        
        X = x_real - mid_pt;
        Xl = x_left - mid_pt;
        Xr = x_right - mid_pt;
        
        tempF = v/Dt * -1.0 * ppi * (7.0/320.0 *cos(ppi *X/20.0) + 7.0/640.0 *cos(ppi *X/10.0) + 1/320 *cos(3.0*ppi *X/20.0) + 1.0/2560.0 *cos(ppi* X/5.0)) * 8.0;

        // potential at x_real
        V = v/Dt * (7.0/16.0 *sin(ppi*X/20.0) + 7.0/64.0 *sin(ppi*X/10.0) + 1.0/48.0 *sin(3.0*ppi*X/20.0) + 1.0/512.0 *sin(ppi*X/5.0)) * 8.0;
        // potential at x_left
        V_left = v/Dt * (7.0/16.0 *sin(ppi*Xl/20.0) + 7.0/64.0 *sin(ppi*Xl/10.0) + 1.0/48.0 *sin(3.0*ppi*Xl/20.0) + 1.0/512.0 *sin(ppi*Xl/5.0)) * 8.0;
        // potential at x_right
        V_right = v/Dt * (7.0/16.0 *sin(ppi*Xr/20.0) + 7.0/64.0 *sin(ppi*Xr/10.0) + 1.0/48.0 *sin(3.0*ppi*Xr/20.0) + 1.0/512.0 *sin(ppi*Xr/5.0)) * 8.0;

        
        F[tid] = tempF;
        VL[tid] = V_left;
        VC[tid] = V;
        VR[tid] = V_right;
        
        
    }
}
        


__global__ void init_W_Cv(double *VL, double *VC, double *VR, double *Wlp, double *Wlm, double *Wrp, double *Wrm, const int N, const double dt, const double dl, const double Dt, const double v){
    
    const int tid = threadIdx.x + blockDim.x * blockIdx.x ;
    double V_left, V, V_right;

  
    if(tid < N) {
    
        V_left = VL[tid];
        V = VC[tid];
        V_right = VR[tid];
        
        double p_sum_p; double p_sum_m;
        
        Wrp[tid] = dt / dl * (v + Dt * (V - V_right)/dl) / (1 - exp(-(v / Dt + (V - V_right)/dl) * dl));
        Wrm[tid] = dt / dl * (-v + Dt * (V - V_right)/dl) / (1 - exp(-(-v / Dt + (V - V_right)/dl) * dl));
        
        Wlp[tid] = -dt / dl * (v + Dt * (V_left - V)/dl) / (1 - exp((v / Dt + (V_left - V)/dl) * dl));
        Wlm[tid] = -dt / dl * (-v + Dt * (V_left - V)/dl) / (1 - exp((-v / Dt + (V_left - V)/dl) * dl));
        

        p_sum_p = Wrp[tid] + Wlp[tid];
        p_sum_m = Wrm[tid] + Wlm[tid];

        if(p_sum_p > 1){
            Wlp[tid] /= p_sum_p;
            Wrp[tid] /= p_sum_p;
        }
        
        if(p_sum_m > 1){
            Wlm[tid] /= p_sum_m;
            Wrm[tid] /= p_sum_m;
        }
    }
}


__global__ void init_W_C0pot(double *VL, double *VC, double *VR, double *Wlp, double *Wlm, double *Wrp, double *Wrm, const int N, const double dt, const double dl, const double Dt, const double v){
    
    const int tid = threadIdx.x + blockDim.x * blockIdx.x ;
    double V_right, V_left, V;

  
    if(tid < N) {
    
        V_right = VR[tid];
        V_left = VL[tid];
        V = VC[tid];
        
        double p_sum_p; double p_sum_m;
        
        Wrp[tid] = dt / dl / dl * Dt * exp(0.5 * (v / Dt) * dl - 0.5 * (V_right - V));
        Wrm[tid] = dt / dl / dl * Dt * exp(0.5 * (-v / Dt) * dl - 0.5 * (V_right - V));
        
        Wlp[tid] = dt / dl / dl * Dt * exp(-0.5 * (v / Dt) * dl - 0.5 * (V_left - V));
        Wlm[tid] = dt / dl / dl * Dt * exp(-0.5 * (-v / Dt) * dl - 0.5 * (V_left - V));
        

        p_sum_p = Wrp[tid] + Wlp[tid];
        p_sum_m = Wrm[tid] + Wlm[tid];

        if(p_sum_p > 1){
            Wlp[tid] /= p_sum_p;
            Wrp[tid] /= p_sum_p;
        }
        
        if(p_sum_m > 1){
            Wlm[tid] /= p_sum_m;
            Wrm[tid] /= p_sum_m;
        }
    }
}


__global__ void update(double *pp, double *pm, double *Wlp, double *Wlm, double *Wrp, double *Wrm, double Wflip, const int N){
const int tid = threadIdx.x + blockDim.x * blockIdx.x ;
    if(tid < N) {
   
              
        int left; int right;
        left = (N + tid - 1) % N;
        right = (N + tid + 1) % N;
        
        double ppl; double ppr; double pml; double pmr;
        double new_pp; double new_pm;
        
        ppl = pp[left];
        ppr = pp[right];
        pml = pm[left];
        pmr = pm[right];
        
        new_pp = Wrp[left] * ppl + Wlp[right] * ppr + (1 - Wrp[tid] - Wlp[tid]) * pp[tid];
        new_pm = Wrm[left] * pml + Wlm[right] * pmr + (1 - Wrm[tid] - Wlm[tid]) * pm[tid];
     
        pp[tid] = Wflip * new_pm + (1 - Wflip) * new_pp;
        pm[tid] = Wflip * new_pp + (1 - Wflip) * new_pm;
            
        //pp[tid] = new_pp;
        //pm[tid] = new_pm;

        
    }
}


__global__ void EP(double *pp, double *pm, double *VL, double *VC, double *VR, double *Wlp, double *Wlm, double *Wrp, double *Wrm, double *EPp, double *EPm, const int N, const double dt, const double dl, const double Dt, const double v){
const int tid = threadIdx.x + blockDim.x * blockIdx.x ;
double V_left, V, V_right;   
    if(tid < N) {
        V_left = VL[tid];
        V = VC[tid];
        V_right = VR[tid];
        
        EPp[tid] = pp[tid] * (Wlp[tid] * (-(v / Dt) * dl / dt - (V_left - V) / dt) + Wrp[tid] * (+(v / Dt) * dl / dt - (V_right - V) / dt));
        EPm[tid] = pm[tid] * (Wlm[tid] * (-(-v / Dt) * dl / dt - (V_left - V) / dt) + Wrm[tid] * (+(-v / Dt) * dl / dt - (V_right - V) / dt));
        
    }
}
        

void save_current_state(double *pp, double *pm, double *EPp, double *EPm, const int N, char *filename){
        FILE *fp = fopen(filename, "w");
        if (fp == NULL) {printf("Error opening the file %s", filename); return;}
        
        for (int i = 0; i < N; i++) {
                fprintf(fp, "%.15e %.15e %.15e %.15e ", pp[i], pm[i], EPp[i], EPm[i]);
        }
        fprintf(fp, "\n");

        fclose(fp);
        
}
