#include <math.h>
#include <unistd.h>
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
#include <time.h>

#define two_ppi (6.28318530717958648)
#define ppi (3.14159265358979324)

#ifndef max
#define max(a,b)  (((a) > (b)) ? (a) : (b))
#endif

#ifndef min
#define min(a,b)  (((a) < (b)) ? (a) : (b))
#endif

struct ptcl {
    int x;   // position
    int s; // direction of self-propulsion
    
    double even_EP;     // Entropy production for even parity
    double J;
} ;


void error_output(const char *desc)
{
    printf("%s\n", desc) ; exit(-1) ;
}

double rand_unif() {
     // <- main으로 옮기기
    double x = ((double)rand()) / ((double)RAND_MAX);
    
    if (x == 0) {
        while (x == 0) {
            x = ((double)rand()) / ((double)RAND_MAX);
        }
    }
    
    return x;
}
    
    

// initializing RNG for all threads with the same seed
// each state setup will be the state after 2^{67}*tid calls 
__global__ void initialize_prng(const int ptclNum, unsigned int seed, curandState *state)
{
    const int tid = threadIdx.x + blockDim.x * blockIdx.x ;
    if(tid<ptclNum)
        curand_init(seed, tid, 0, &state[tid]) ;
}


void init_random_config_host(const int ptclNum, const double Lsize, const double dl, struct ptcl *hostPtcl)
{
    for (int i = 0; i < ptclNum; i++){
        hostPtcl[i].x = (int) (rand_unif() * (Lsize / dl));
        hostPtcl[i].s = (int)(rand_unif()*2)*2 - 1;
        hostPtcl[i].even_EP = 0.0;
        hostPtcl[i].J = 0.0;
    } 
}

__global__ void init_random_config_dev(curandState *state, const int ptclNum)
{
    const int tid = threadIdx.x + blockDim.x * blockIdx.x ;
    if(tid<ptclNum){
        curandState localState = state[tid];
        state[tid] = localState;
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
        
        Wrp[tid] = dt / dl / dl * (v * dl - Dt * (V_right - V)) / (1 - exp(-(v / Dt) * dl + (V_right - V)));
        Wrm[tid] = dt / dl / dl * (-v * dl - Dt * (V_right - V)) / (1 - exp(-(-v / Dt) * dl + (V_right - V)));
        
        Wlp[tid] = dt / dl / dl * (v * -dl - Dt * (V_left - V)) / (1 - exp(-(v / Dt) * -dl + (V_left - V)));
        Wlm[tid] = dt / dl / dl * (-v * -dl - Dt * (V_left - V)) / (1 - exp(-(-v / Dt) * -dl + (V_left - V)));
        

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
    double V_left, V, V_right;

  
    if(tid < N) {
    
        V_left = VL[tid];
        V = VC[tid];
        V_right = VR[tid];
        
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



__global__ void move(curandState *state, const double Lsize, const int ptclNum, const double v, const double dt, const double dl, const double Dt, const double alpha, double *VL, double *VC, double *VR, double *Wlp, double *Wlm, double *Wrp, double *Wrm, struct ptcl *devPtcl)
{
    const int tid = threadIdx.x + blockDim.x * blockIdx.x ;
    
    double vx;
    int x, s;
    int sys_size = (int)(Lsize/dl);
    double V, V_left, V_right; // V is potential normalized by temperature.

    // size of the leftside and rightside of the potential
        
    if(tid<ptclNum){
               
        double prr, prl, p_sum;
        x = devPtcl[tid].x;
        s = devPtcl[tid].s;
        vx = v * devPtcl[tid].s;
        
        if(s == 1){
            prr = Wrp[x];
            prl = Wlp[x];
        }
        else{
            prr = Wrm[x];
            prl = Wlm[x];
        }
        p_sum = prr + prl;
        

        if(p_sum > 1){
            prr /= p_sum;
            prl /= p_sum;
        }
        
        //  flip the particle
        double rn1 = curand_uniform(&state[tid]);
        if (rn1 < 0.5 - 0.5*exp(-alpha * dt)){
            devPtcl[tid].s *= -1;
        }

        // move the particle
        double rn = curand_uniform(&state[tid]);
        
        // Define V_left, V, V_Right
        V_left = VL[x];
        V = VC[x];
        V_right = VR[x];
        
        if (rn < prl){
            devPtcl[tid].x = (x - 1 + sys_size)%sys_size; //go left
            devPtcl[tid].even_EP += -(vx / Dt) * dl - (V_left - V);
            devPtcl[tid].J += -dl;
        }
        else if (rn < prl + prr){
            devPtcl[tid].x = (x + 1)%sys_size; //go right
            devPtcl[tid].even_EP += +(vx / Dt) * dl - (V_right - V);
            devPtcl[tid].J += dl;
        }
        else {
            devPtcl[tid].x = x;
            devPtcl[tid].even_EP += 0.0;
            devPtcl[tid].J += 0.0;
        }
    }
}

__global__ void reset(struct ptcl *devPtcl){
    const int tid = threadIdx.x + blockDim.x * blockIdx.x ;
    
    devPtcl[tid].even_EP = 0.0;
    devPtcl[tid].J = 0.0;
}

