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

struct cell {
    float potR, potG;
    float angle;
    int type; // 0 for empty, 1 for red, -1 for green
    int can; // candidate for move
    
    int index;
    int nb;
    int findex;
    
    int dx; // current x position - former x position
    int dy; // current y position - former y position
    float fangle; // angle for former step
    
    float even_EP;     // Entropy production for even parity
    float odd_EP;      // Entropy production for odd parity
} ;


void error_output(const char *desc)
{
    printf("%s\n", desc) ; exit(-1) ;
}

float rand_unif() {
     // <- main으로 옮기기
    float x = ((float)rand()) / ((float)RAND_MAX);
    
    if (x == 0) {
        while (x == 0) {
            x = ((float)rand()) / ((float)RAND_MAX);
        }
    }
    
    return x;
}
    
    

// initializing RNG for all threads with the same seed
// each state setup will be the state after 2^{67}*tid calls 
__global__ void initialize_prng(const int Lxsize, const int Lysize,
        unsigned int seed, curandState *state)
{
    const int tid = threadIdx.x + blockDim.x * blockIdx.x ;
    if(tid<Lxsize*Lysize)
        curand_init(seed, tid, 0, &state[tid]) ;
}


void init_random_config_host(const int Lxsize, const int Lysize, const int ptlsNum_r, const int ptlsNum_g, struct cell *hostLattice)
{
    int *temp;
    temp = (int *)malloc(sizeof(int) * Lxsize * Lysize);
    for (int i = 0; i < Lxsize * Lysize; i++){
        temp[i] = i;
        hostLattice[i].potR = 0.0;
        hostLattice[i].potG = 0.0;
        hostLattice[i].angle = 0.0;
        hostLattice[i].type = 0;
        hostLattice[i].findex = -2;
        hostLattice[i].can = -2;
        hostLattice[i].index = -2;
        hostLattice[i].nb = 0;
        
        hostLattice[i].dx = 0;
        hostLattice[i].dy = 0;
        hostLattice[i].fangle = 0.0;

        hostLattice[i].even_EP = 0.0;
        hostLattice[i].odd_EP = 0.0;
    }
    int m = 0;
    
    for (int i = 0; i < ptlsNum_r; i++){
        int tt = i + (int) (rand_unif() * (Lxsize*Lysize - i));
        hostLattice[temp[tt]].type = 1;
        hostLattice[temp[tt]].index = m;
        hostLattice[temp[tt]].findex = m; m++;
        hostLattice[temp[tt]].angle = rand_unif() * two_ppi - ppi;
        
        int ttt = temp[i];
        temp[i] = temp[tt];
        temp[tt] = ttt;
    }
    
    for (int i = ptlsNum_r; i < ptlsNum_r + ptlsNum_g; i++){
        int tt = i + (int) (rand_unif() * (Lxsize*Lysize - i));
        hostLattice[temp[tt]].type = -1;
        hostLattice[temp[tt]].index = m;
        hostLattice[temp[tt]].findex = m; m++;
        hostLattice[temp[tt]].angle = rand_unif() * two_ppi - ppi;
        
        int ttt = temp[i];
        temp[i] = temp[tt];
        temp[tt] = ttt;
    }
    
}

__global__ void init_random_config_dev(curandState *state, const int Lxsize, const int Lysize)
{
    const int tid = threadIdx.x + blockDim.x * blockIdx.x ;
    if(tid<Lxsize*Lysize){
        curandState localState = state[tid];
        state[tid] = localState;
    }
}


__global__ void potential_cal(const int Lxsize, const int Lysize, const int ptlsNum, const float F_rep, const float F_adh_r, const float F_adh_g, const float dl, const float vr, const float vg, const float Dt, struct cell *devLattice)
{
    const int tid = threadIdx.x + blockDim.x * blockIdx.x ;
    if(tid<Lxsize*Lysize){
        
        devLattice[tid].findex = devLattice[tid].index;
        
        float temp_R = 0.0;
        float temp_G = 0.0;
        
        temp_R += 1000.0 * (devLattice[tid].type) * (devLattice[tid].type);  // 1000 is large number to avoid overlapping
        temp_G += 1000.0 * (devLattice[tid].type) * (devLattice[tid].type);  // 1000 is large number to avoid overlapping
        
        
        devLattice[tid].potR = temp_R;
        devLattice[tid].potG = temp_G;
    }
}


__global__ void move_can(curandState *state, const int Lxsize, const int Lysize, const int ptlsNum, const float F_rep, const float F_adh_r, const float F_adh_g, const float vr, const float vg, const float dt, const float dl, const float Dt, struct cell *devLattice, int which_sub)
{
    const int tid = threadIdx.x + blockDim.x * blockIdx.x ;
    int x = tid%Lxsize;
    int y = tid/Lxsize;  
    
    if(tid<Lxsize*Lysize && (3*y + x) % 5 == which_sub){
        
        float pot_self, angle_self, vx, vy, dHu, dHd, dHr, dHl;
        
        // empty, R, G
        
        if(devLattice[tid].type == 1){
            pot_self = devLattice[tid].potR - 1000.0; // minus to exclude self interaction
            angle_self = devLattice[tid].angle;
            vx = vr * cosf(angle_self);
            vy = vr * sinf(angle_self);

            dHu = devLattice[x + ((y+1 + Lysize)%Lysize)*Lxsize].potR - pot_self; // minus to exclude self interaction
            dHd = devLattice[x + ((y-1 + Lysize)%Lysize)*Lxsize].potR - pot_self;
            dHr = devLattice[(x+1 + Lxsize)%Lxsize + y*Lxsize].potR - pot_self;
            dHl = devLattice[(x-1 + Lxsize)%Lxsize + y*Lxsize].potR - pot_self;
            
        }
        else if(devLattice[tid].type == -1){
            pot_self = devLattice[tid].potG - 1000.0;
            angle_self = devLattice[tid].angle;
            vx = vg * cosf(angle_self);
            vy = vg * sinf(angle_self);

            dHu = devLattice[x + ((y+1 + Lysize)%Lysize)*Lxsize].potG - pot_self; // minus to exclude self interaction
            dHd = devLattice[x + ((y-1 + Lysize)%Lysize)*Lxsize].potG - pot_self;
            dHr = devLattice[(x+1 + Lxsize)%Lxsize + y*Lxsize].potG - pot_self;
            dHl = devLattice[(x-1 + Lxsize)%Lxsize + y*Lxsize].potG - pot_self; 
        }
        else if(devLattice[tid].type == 0){
            devLattice[tid].can = -2;
            return;
        }
        
        float pru, prd, prr, prl;
        float o_pru, o_prd, o_prr, o_prl;

        if (dHu < 500){ 
            if(vy!=0.0){
                pru = dt / dl * vy / (1.0 - expf(- vy * dl / Dt)); 
                o_pru = - dt / dl * vy / (1.0 - expf(+ vy * dl / Dt)); 
            }
            else{
                pru = Dt*dt/dl/dl;
                o_pru = Dt*dt/dl/dl;
            }
        }
        else { pru = 0.0; o_pru = 0.0; }
        if (dHd < 500){ 
            if(vy!=0.0){
                prd = dt / dl * vy / (expf(vy * dl / Dt) - 1.0);
                o_prd = - dt / dl * vy / (expf(- vy * dl / Dt) - 1.0);
            }
            else{
                prd = Dt*dt/dl/dl;
                o_prd = Dt*dt/dl/dl;
            }
        }
        else { prd = 0.0; o_prd = 0.0; } 
        if (dHr < 500){
            if(vx!=0.0){
                prr = dt / dl * vx / (1.0 - expf(- vx * dl / Dt)); 
                o_prr = - dt / dl * vx / (1.0 - expf(+ vx * dl / Dt)); 
            }
            else{
                prr = Dt*dt/dl/dl;
                o_prr = Dt*dt/dl/dl;
            }
        }
        else { prr = 0.0; o_prr = 0.0; }
        if (dHl < 500){ 
            if(vx!=0.0){
                prl = dt / dl * vx / (expf(vx * dl / Dt) - 1.0); 
                o_prl = - dt / dl * vx / (expf(- vx * dl / Dt) - 1.0); 
            }
            else{
                prl = Dt*dt/dl/dl;
                o_prl = Dt*dt/dl/dl;
            }
        }
        else { prl = 0.0; o_prl = 0.0;}
        
        
        float rn = curand_uniform(&state[tid]);
        
        // devLattice[tid].can = 1;
        
        if (rn < prl){
            devLattice[tid].can = (x-1 + Lxsize)%Lxsize + y*Lxsize; //go left
            devLattice[tid].even_EP = - vx*dl/Dt;
        }
        else if (rn < prl + prr){
            devLattice[tid].can = (x+1 + Lxsize)%Lxsize + y*Lxsize; //go right
            devLattice[tid].even_EP = + vx*dl/Dt;
        }
        else if (rn < prl + prr + pru){
            devLattice[tid].can = x + ((y+1 + Lysize)%Lysize)*Lxsize; //go up
            devLattice[tid].even_EP = + vy*dl/Dt;
        }
        else if (rn < prl + prr + pru + prd){
            devLattice[tid].can = x + ((y-1 + Lysize)%Lysize)*Lxsize; //go down
            devLattice[tid].even_EP = - vy*dl/Dt;
        }
        else {
            devLattice[tid].can = tid;
            devLattice[tid].odd_EP = logf((1.0-(pru + prd + prr + prl)) / (1.0-(o_pru + o_prd + o_prr + o_prl)));
        }
    }
}

__global__ void rotate(curandState *state, const int Lxsize, const int Lysize, const int ptlsNum, const float K, const float D, const float dt, struct cell *devLattice)
{
    const int tid = threadIdx.x + blockDim.x * blockIdx.x ;
    
    if(tid<Lxsize*Lysize && devLattice[tid].type != 0){
                
        devLattice[tid].fangle = devLattice[tid].angle; //save the former angle to fangle before update
        
        devLattice[tid].angle += + pow(D * dt, 0.5) * (2.*curand_uniform(&state[tid])-1.0);
        
        if(devLattice[tid].angle < -1. * ppi)
            devLattice[tid].angle += two_ppi;
        if(devLattice[tid].angle >ppi)
            devLattice[tid].angle -= two_ppi;
    }
}

__global__ void candidate(const int Lxsize, const int Lysize, const int ptlsNum, struct cell *devLattice, int which_sub)
{
    const int tid = threadIdx.x + blockDim.x * blockIdx.x ;
    int x = tid%Lxsize;
    int y = tid/Lxsize;  
    
    if(tid<Lxsize*Lysize && (3*y + x) % 5 == which_sub){
        if (devLattice[devLattice[tid].can].type != 0) { devLattice[tid].can = tid; }
        
    }
}

__global__ void copyLattice(const int Lxsize, const int Lysize, struct cell *devLattice, struct cell *tempdevLattice)
{
    const int tid = threadIdx.x + blockDim.x * blockIdx.x ;
    if(tid<Lxsize*Lysize){
        
        // copy to tempdev
        tempdevLattice[tid].type = devLattice[tid].type;
        tempdevLattice[tid].angle = devLattice[tid].angle;
        tempdevLattice[tid].index = devLattice[tid].index;

        tempdevLattice[tid].fangle = devLattice[tid].fangle;
        tempdevLattice[tid].dx = devLattice[tid].dx;
        tempdevLattice[tid].dy = devLattice[tid].dy;

        tempdevLattice[tid].even_EP = devLattice[tid].even_EP;
        tempdevLattice[tid].odd_EP = devLattice[tid].odd_EP;
    }
}


__global__ void exchange_cell(const int Lxsize, const int Lysize, struct cell *devLattice, struct cell *tempdevLattice, int which_sub)
{
    const int tid = threadIdx.x + blockDim.x * blockIdx.x ;
    if(tid<Lxsize*Lysize){
        
        devLattice[tid].dx = 0; // set dx=dy=0 unless a particle move to the position of given tid.
        devLattice[tid].dy = 0;
        int x = tid%Lxsize;
        int y = tid/Lxsize;
        
        // When we select updating sublattice, if the site is filled and its candidate is not itself, then delete the site. If not, do nothing
        if ( (3*y + x) % 5 == which_sub && devLattice[tid].type != 0){
            if ( devLattice[tid].can != tid ){ 
                devLattice[tid].type = 0;
                // devLattice[tid].can = -2; 
            }
        }
        
        
        if ( (3*y + x) % 5 != which_sub && devLattice[tid].type == 0){
            int left = (x-1 + Lxsize)%Lxsize + y * Lxsize;
            int right = (x+1 + Lxsize)%Lxsize + y * Lxsize;
            int up = x + (y + 1 + Lysize)%Lysize * Lxsize;
            int down = x + (y - 1 + Lysize)%Lysize * Lxsize;
            
            if ( devLattice[left].type != 0 && (3*y + (x-1 + Lxsize)%Lxsize) % 5 == which_sub){
                if ( devLattice[left].can == tid ){
                    devLattice[tid].type = tempdevLattice[left].type;
                    devLattice[tid].angle = tempdevLattice[left].angle;
                    devLattice[tid].even_EP = tempdevLattice[left].even_EP;
                    devLattice[tid].odd_EP = tempdevLattice[left].odd_EP;
                }
            }
            
            else if ( devLattice[right].type != 0 && (3*y + (x+1 + Lxsize)%Lxsize) % 5 == which_sub){
                if ( devLattice[right].can == tid ){
                    devLattice[tid].type = tempdevLattice[right].type;
                    devLattice[tid].angle = tempdevLattice[right].angle;
                    devLattice[tid].even_EP = tempdevLattice[right].even_EP;
                    devLattice[tid].odd_EP = tempdevLattice[right].odd_EP;
                }
            }
            
            else if ( devLattice[up].type != 0 && (3*(y + 1 + Lysize)%Lysize + x) % 5 == which_sub){
                if ( devLattice[up].can == tid ){
                    devLattice[tid].type = tempdevLattice[up].type;
                    devLattice[tid].angle = tempdevLattice[up].angle;
                    devLattice[tid].even_EP = tempdevLattice[up].even_EP;
                    devLattice[tid].odd_EP = tempdevLattice[up].odd_EP;
                }
            }
            
            else if ( devLattice[down].type != 0 && (3*(y - 1 + Lysize)%Lysize + x) % 5 == which_sub){
                if ( devLattice[down].can == tid ){
                    devLattice[tid].type = tempdevLattice[down].type;
                    devLattice[tid].angle = tempdevLattice[down].angle;
                    devLattice[tid].even_EP = tempdevLattice[down].even_EP;
                    devLattice[tid].odd_EP = tempdevLattice[down].odd_EP;
                }
            }
            
        }
         
    }
}


__global__ void delete_cell(const int Lxsize, const int Lysize, struct cell *devLattice, struct cell *tempdevLattice, int which_sub)
{
    const int tid = threadIdx.x + blockDim.x * blockIdx.x ;
    if(tid<Lxsize*Lysize){
        
        devLattice[tid].dx = 0; // set dx=dy=0 unless a particle move to the position of given tid.
        devLattice[tid].dy = 0;
        int x = tid%Lxsize;
        int y = tid/Lxsize;
        
        // When we select updating sublattice, if the site is filled and its candidate is not itself, then delete the site. If not, do nothing
        if ( (3*y + x) % 5 == which_sub && devLattice[tid].type != 0){
            if ( devLattice[tid].can != tid ){ 
                devLattice[tid].type = 0;
                // devLattice[tid].can = -2; 
            }
        }
        
         
    }
}

__global__ void create_cell(const int Lxsize, const int Lysize, struct cell *devLattice, struct cell *tempdevLattice, int which_sub)
{
    const int tid = threadIdx.x + blockDim.x * blockIdx.x ;
    if(tid<Lxsize*Lysize){
        
        devLattice[tid].dx = 0; // set dx=dy=0 unless a particle move to the position of given tid.
        devLattice[tid].dy = 0;
        int x = tid%Lxsize;
        int y = tid/Lxsize;
        
        
        if ( (3*y + x) % 5 != which_sub && devLattice[tid].type == 0){
            int left = (x-1 + Lxsize)%Lxsize + y * Lxsize;
            int right = (x+1 + Lxsize)%Lxsize + y * Lxsize;
            int up = x + (y + 1 + Lysize)%Lysize * Lxsize;
            int down = x + (y - 1 + Lysize)%Lysize * Lxsize;
            
            if ( devLattice[left].type != 0 && (3*y + (x-1 + Lxsize)%Lxsize) % 5 == which_sub){
                if ( devLattice[left].can == tid ){
                    devLattice[tid].type = tempdevLattice[left].type;
                    devLattice[tid].angle = tempdevLattice[left].angle;
                    devLattice[tid].even_EP = tempdevLattice[left].even_EP;
                    devLattice[tid].odd_EP = tempdevLattice[left].odd_EP;
                }
            }
            
            else if ( devLattice[right].type != 0 && (3*y + (x+1 + Lxsize)%Lxsize) % 5 == which_sub){
                if ( devLattice[right].can == tid ){
                    devLattice[tid].type = tempdevLattice[right].type;
                    devLattice[tid].angle = tempdevLattice[right].angle;
                    devLattice[tid].even_EP = tempdevLattice[right].even_EP;
                    devLattice[tid].odd_EP = tempdevLattice[right].odd_EP;
                }
            }
            
            else if ( devLattice[up].type != 0 && (3*(y + 1 + Lysize)%Lysize + x) % 5 == which_sub){
                if ( devLattice[up].can == tid ){
                    devLattice[tid].type = tempdevLattice[up].type;
                    devLattice[tid].angle = tempdevLattice[up].angle;
                    devLattice[tid].even_EP = tempdevLattice[up].even_EP;
                    devLattice[tid].odd_EP = tempdevLattice[up].odd_EP;
                }
            }
            
            else if ( devLattice[down].type != 0 && (3*(y - 1 + Lysize)%Lysize + x) % 5 == which_sub){
                if ( devLattice[down].can == tid ){
                    devLattice[tid].type = tempdevLattice[down].type;
                    devLattice[tid].angle = tempdevLattice[down].angle;
                    devLattice[tid].even_EP = tempdevLattice[down].even_EP;
                    devLattice[tid].odd_EP = tempdevLattice[down].odd_EP;
                }
            }
            
        }
         
    }
}


