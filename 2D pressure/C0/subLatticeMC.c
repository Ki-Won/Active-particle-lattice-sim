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
    int y;
    double theta; // direction of self-propulsion
    double force;     // Entropy production for even parity
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


void init_random_config_host(const int ptclNum, const double Lxsize, const double Lysize, const double dl, struct ptcl *hostPtcl)
{
    for (int i = 0; i < ptclNum; i++){
        hostPtcl[i].x = (int) (rand_unif() * (Lxsize / dl));
        hostPtcl[i].y = (int) (rand_unif() * (Lysize / dl));
        hostPtcl[i].theta = rand_unif() * two_ppi;
        hostPtcl[i].force = 0.0;
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


__global__ void move(curandState *state, const double Lxsize, const double Lysize, const int ptclNum, const double v, const double dt, const double dl, const double Dt, const double Dr, struct ptcl *devPtcl)
{
    const int tid = threadIdx.x + blockDim.x * blockIdx.x ;
    
    double vx, vy; 
    double F;
    double V, V_left, V_right; //V is potential normalized by T
    int x, y;
    
    double x_real, x_left, x_right; //actual position not grid coordinate

    double F0 = 0.5 * v / Dt;
    double rpot_loc = Lxsize - 40;
        
    if(tid<ptclNum){
        
        x = devPtcl[tid].x;
        y = devPtcl[tid].y;
        x_real = (double)(x)*dl;
        x_left = x_real - dl;    
        x_right = x_real + dl;
        
        vx = v * cos(devPtcl[tid].theta);
        vy = v * sin(devPtcl[tid].theta);

        // calculating normalized force and normalized potential for given position
        if(x_real <= 0.0){
            F = F0 * pow(-1 * x_real,7);
            V = F0 * pow(x_real,8)/8;
        }
        else if(x_real >= rpot_loc){
            F = - F0 * pow(x_real-rpot_loc,7);
            V = F0 * pow(x_real-rpot_loc,8)/8;
        }
        else{
            F = 0.0;
            V = 0.0;
        }
        // calculating potential at the left 
        if(x_left <= 0.0){
            V_left = F0 * pow(x_left,8)/8;
        }
        else if(x_real >= rpot_loc){
            V_left = F0 * pow(x_left-rpot_loc,8)/8;
        }
        else{
            V_left = 0.0;
        }
        // calculating potential at the right
        if(x_right <= 0.0){
            V_right = F0 * pow(x_right,8)/8;
        }
        else if(x_real >= rpot_loc){
            V_right = F0 * pow(x_right-rpot_loc,8)/8;
        }
        else{
            V_right = 0.0;
        }
        
        double prr, prl, pru, prd, p_sum;
        
        if(x_real >= Lxsize - 20.0){
            prr = 0.0;
        }
        else{ 
            prr = dt / dl / dl * Dt * exp(0.5 * (vx / Dt) * dl - 0.5 * (V_right - V));
        }
        
        if(x_real <= -20.0){
            prl = 0.0;
        }
        else{ 
            prl = dt / dl / dl * Dt * exp(-0.5 * (vx / Dt) * dl - 0.5 * (V_left - V));
        }

        pru = dt / dl / dl * Dt * exp(0.5 * (vy / Dt) * dl);
        prd = dt / dl / dl * Dt * exp(-0.5 * (vy / Dt) * dl);

        p_sum = prr + prl + pru + prd;

        if(p_sum > 1){
            prr /= p_sum;
            prl /= p_sum;
            pru /= p_sum;
            prd /= p_sum;
        }
        
        //  flip the particle
        devPtcl[tid].theta += + pow(6 * Dr * dt, 0.5) * (2.*curand_uniform(&state[tid])-1.0);
        
        if(devPtcl[tid].theta < -1. * ppi)
            devPtcl[tid].theta += two_ppi;
        if(devPtcl[tid].theta > ppi)
            devPtcl[tid].theta -= two_ppi;

        
        // move the particle
        double rn = curand_uniform(&state[tid]);
        
        if (rn < prl){
            devPtcl[tid].x = x - 1; //go left
            devPtcl[tid].force = fabs(F);
        }
        else if (rn < prl + prr){
            devPtcl[tid].x = x + 1; //go right
            devPtcl[tid].force = fabs(F);
        }
        else if (rn < prl + prr + pru){
            devPtcl[tid].y = y + 1; //go up
            devPtcl[tid].force = fabs(F);
        }
        else if (rn < prl + prr + pru + prd){
            devPtcl[tid].y = y - 1; //go down
            devPtcl[tid].force = fabs(F);
        }
        else {
            devPtcl[tid].force = fabs(F);
        }
    }
}


