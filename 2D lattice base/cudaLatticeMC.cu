#define R0_r (0.7)
#define MaxThreads (1024)

#include "subLatticeMC.c" 
#include <sys/stat.h>
#include <string.h>
#include <time.h>

void save_current_state(struct cell *hostLattice, const int Lxsize, const int Lysize, char *filename);
__global__ void field_FT(struct cell *devLattice, const int Lxsize, const int Lysize, float *rho_tilde_Re, float *rho_tilde_Im);


int main(int argc, char *argv[])
{
        
        // device setting
        const int    device_num = atoi(argv[1]);
        if(device_num<0 || device_num>3) error_output("invalid device number") ;
        cudaSetDevice(device_num);
        
        // folder setting
        char folder_name[100];
        sprintf(folder_name, "%s", argv[2]);
        mkdir(folder_name, S_IRWXU);

     
        // Lattice and cell setting
        const int    Lx_size  = atoi(argv[3]);
        const int    Ly_size  = atoi(argv[4]);
        const int    tmax   = atoi(argv[5]);

        // Dynamics setting
        const float  dt     = 0.5;
        const float  dl     = 1.0;
        const float  rho_r    = atof(argv[6]);
        const float  rho_g    = atof(argv[7]);
        const float  speed_r  = 1.0;
        const float  speed_g  = 0.08;
        const float  F_rep = 0.0;
        const float  F_adh_r = 0.0;
        const float  F_adh_g = 0.0;
        
        // const float  F_wall = 400.0;
        const float  K      = 0.0; // Vicsek alignmetn strength
        const float  D      = 0.01; // angular noise
        const float  Dt     = 0.4; // translational diffusion coefficient

        // time measurement for initialization
        clock_t start, end;
        // time measurement for total simulation
        clock_t start1, end1;
        //float res;

        start = clock();
        
        srand(time(NULL));


        // total number of particles
        const  int   ptlsNum_r = (int)(Lx_size*Ly_size*rho_r) ;
        const  int   ptlsNum_g = (int)(Lx_size*Ly_size*rho_g) ;
        const  int   ptlsNum   = ptlsNum_r + ptlsNum_g ;

        
        // grid dimension
        const int nThreads = (MaxThreads<Lx_size*Ly_size)? MaxThreads : Lx_size*Ly_size;
        const int nBlocks  = (Lx_size*Ly_size+nThreads-1)/nThreads; 

        printf("%d %d\n", nThreads, nBlocks);
        
        size_t memSize = sizeof(struct cell) * Lx_size*Ly_size;
        
        struct cell *initLattice;
        struct cell *hostLattice;
        struct cell *temphostLattice;
        struct cell *rollbackLattice;

        initLattice = (struct cell *)malloc(sizeof(struct cell) * Lx_size * Ly_size);
        hostLattice = (struct cell *)malloc(sizeof(struct cell) * Lx_size * Ly_size);
        temphostLattice = (struct cell *)malloc(sizeof(struct cell) * Lx_size * Ly_size);
        
        rollbackLattice = (struct cell *)malloc(sizeof(struct cell) * Lx_size * Ly_size);
        
        float *devVangle;
        cudaMalloc(&devVangle, sizeof(float)*Lx_size*Ly_size);

        
        std::random_device rd;
        unsigned int seed = rd();
        // seed = 1234;
        // initialize the PRNGs
        curandState *devStates ;
        cudaMalloc((void **)&devStates, Lx_size*Ly_size*sizeof(curandState)) ;
        initialize_prng<<<nBlocks, nThreads>>>(Lx_size, Ly_size, seed, devStates) ;
           
        init_random_config_host(Lx_size, Ly_size, ptlsNum_r, ptlsNum_g, initLattice);
        
        int tot_ptls = 0;
        int numA = 0;
        int numB = 0;
        int numC = 0;
        int numD = 0;
        int numE = 0;
        int tot_nbs = 0;
        int check = 0;
        int temp_type = 0;
        int temp_pos = 0;
        float temp_angle = 0;

        for (int i = 0; i < Lx_size; i++){
            for (int j = 0; j < Ly_size; j++){
                //printf("%3d ",initLattice[i + j*Lx_size].type);
                if ( initLattice[i + j*Lx_size].type != 0) { tot_ptls += 1; }
            }
            //printf("\n");
        }
        
        printf("total particle number = %d\n", tot_ptls);
        
        
        float T = 1.0;
        
        struct cell *devLattice ;
        // Cell in the device
        cudaMalloc(&devLattice, sizeof(struct cell)*Lx_size*Ly_size) ;
        
        struct cell *tempdevLattice ;
        cudaMalloc(&tempdevLattice, sizeof(struct cell)*Lx_size*Ly_size) ;
        
        
        // for the length measurement
        float *dev_rho_tilde_Re, *dev_rho_tilde_Im;
        cudaMalloc(&dev_rho_tilde_Re, 201 * 201 * sizeof(float));
        cudaMalloc(&dev_rho_tilde_Im, 201 * 201 * sizeof(float));
        
        float *host_rho_tilde_Re, *host_rho_tilde_Im;
        host_rho_tilde_Re = (float *)malloc(201 * 201 * sizeof(float));
        host_rho_tilde_Im = (float *)malloc(201 * 201 * sizeof(float));


        cudaMemcpy(devLattice, initLattice, memSize, cudaMemcpyHostToDevice);
        cudaMemcpy(rollbackLattice, devLattice, memSize, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        
        end = clock();
        printf("intialize (in sec) : %f\n", (double)(end - start) / CLOCKS_PER_SEC);
        
        
        // reptition start  
        
        start1 = clock();
        
        int t = 0;
        while (t < tmax){
        
            for (int subrep = 0; subrep < 5; subrep++){
            
                int which_sub = (int) (rand_unif() * 5);
                //int which_sub = 0;
            
                potential_cal<<<nBlocks, nThreads>>>(Lx_size, Ly_size, ptlsNum, F_rep, F_adh_r, F_adh_g, dl, speed_r, speed_g, Dt, devLattice);
                move_can<<<nBlocks, nThreads>>>(devStates, Lx_size, Ly_size, ptlsNum, F_rep, F_adh_r, F_adh_g, speed_r, speed_g, dt, dl, Dt, devLattice, which_sub);
                
                cudaDeviceSynchronize();
                
                //candidate<<<nBlocks, nThreads>>>(Lx_size, Ly_size, ptlsNum, devLattice, which_sub);
                copyLattice<<<nBlocks, nThreads>>>(Lx_size, Ly_size, devLattice, tempdevLattice);
                
                cudaDeviceSynchronize();
                //cudaMemcpy(temphostLattice, devLattice, memSize, cudaMemcpyDeviceToHost);
                //cudaMemcpy(tempdevLattice, temphostLattice, memSize, cudaMemcpyHostToDevice);
               
                
                //exchange_cell<<<nBlocks, nThreads>>>(Lx_size, Ly_size, devLattice, tempdevLattice, which_sub);
                create_cell<<<nBlocks, nThreads>>>(Lx_size, Ly_size, devLattice, tempdevLattice, which_sub);
                cudaDeviceSynchronize();
                delete_cell<<<nBlocks, nThreads>>>(Lx_size, Ly_size, devLattice, tempdevLattice, which_sub);
                
                
            }    
            rotate<<<nBlocks, nThreads>>>(devStates, Lx_size, Ly_size, ptlsNum, K, D, dt, devLattice);    
            cudaMemcpy(hostLattice, devLattice, memSize, cudaMemcpyDeviceToHost);
                
            
                
            
                if (t%250 == 0){
                
                    //cudaMemcpy(hostLattice, devLattice, memSize, cudaMemcpyDeviceToHost);
                    char state_filename[200];
                    sprintf(state_filename, "%s/state_t_%d.txt", folder_name, t);
                    save_current_state(hostLattice, Lx_size, Ly_size, state_filename);

                    tot_ptls = 0;
                    numA = 0;
                    numB = 0;
                    numC = 0;
                    numD = 0;
                    numE = 0;
                    

                    for (int i = 0; i < Lx_size; i++){
                        for (int j = 0; j < Ly_size; j++){
                            //printf("%3d ",initLattice[i + j*Lx_size].type);
                            if ( hostLattice[i + j*Lx_size].type != 0) { tot_ptls += 1; }
                            if ( (3*j + i)%5 ==0 && hostLattice[i + j*Lx_size].type != 0) { numA += 1; }
                            if ( (3*j + i)%5 ==1 && hostLattice[i + j*Lx_size].type != 0) { numB += 1; }
                            if ( (3*j + i)%5 ==2 && hostLattice[i + j*Lx_size].type != 0) { numC += 1; }
                            if ( (3*j + i)%5 ==3 && hostLattice[i + j*Lx_size].type != 0) { numD += 1; }
                            if ( (3*j + i)%5 ==4 && hostLattice[i + j*Lx_size].type != 0) { numE += 1; }
                        }
                    }

                    printf("tot %d A %d B %d C %d D %d E %d, t_step=%d\n", tot_ptls,numA,numB,numC,numD,numE,t);

                }
                
                if (t == (int) T || t == tmax){
                
                        char state_filename[200];
                        sprintf(state_filename, "%s/state_t_%d.txt", folder_name, t);
                        save_current_state(hostLattice, Lx_size, Ly_size, state_filename);

                
                        //<<<201,201>>>(devLattice, Lx_size, Ly_size, dev_rho_tilde_Re, dev_rho_tilde_Im);

                        //cudaDeviceSynchronize();
                        
                        //cudaMemcpy(host_rho_tilde_Re, dev_rho_tilde_Re, 201 * 201 * sizeof(float), cudaMemcpyDeviceToHost);
                        //cudaMemcpy(host_rho_tilde_Im, dev_rho_tilde_Im, 201 * 201 * sizeof(float), cudaMemcpyDeviceToHost);
                        
                        //cudaDeviceSynchronize();
                        
                        //char len_filename[200];

                        //sprintf(len_filename, "%s/t_%d.txt", folder_name, t);
                        //FILE *fp = fopen(len_filename, "w");
                        //for (int i = 0; i < 201 * 201; i++){
                        //        fprintf(fp, "%f ", host_rho_tilde_Re[i]);
                        //}
                        //fprintf(fp, "\n");
                        //for (int i = 0; i < 201 * 201; i++){
                        //        fprintf(fp, "%f ", host_rho_tilde_Im[i]);
                        //}
                        //fprintf(fp, "\n");
                        //fclose(fp);


                        if (T < 2048){
                                T *= 2;
                        }
                        else {
                                T *= 1.1;
                        }

                }
            
            t++;
        }
        
        
        
        end1 = clock();
        printf("Total sim time (in sec) : %f\n", (double)(end1 - start1) / CLOCKS_PER_SEC);
        
        free(hostLattice); free(initLattice); free(temphostLattice); free(rollbackLattice);
        
        cudaFree(devLattice); cudaFree(tempdevLattice); cudaFree(devVangle); cudaFree(devStates);
        
        cudaFree(dev_rho_tilde_Re) ; cudaFree(dev_rho_tilde_Im) ; 
        
        free(host_rho_tilde_Re) ; free(host_rho_tilde_Im) ; 
}



void save_current_state(struct cell *hostLattice, const int Lxsize, const int Lysize, char *filename){
        FILE *fp = fopen(filename, "w");
        if (fp == NULL) {printf("Error opening the file %s", filename); return;}
        
        for (int i = 0; i < Lxsize * Lysize; i++) {
                fprintf(fp, "%d %f %f %f ", hostLattice[i].type, hostLattice[i].angle, hostLattice[i].even_EP, hostLattice[i].odd_EP);
        }
        fprintf(fp, "\n");
        fclose(fp);
        
}

__global__ void field_FT(struct cell *devLattice, const int Lxsize, const int Lysize, float *rho_tilde_Re, float *rho_tilde_Im){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    float k_x;
    float k_y;
    float tx, ty;

    k_x = (float) two_ppi/200 * threadIdx.x;
    k_y = (float) two_ppi/200 * blockIdx.x;

    tx = 0;
    ty = 0;

    for (int tid = 0; tid < Lxsize * Lysize; tid++){
        int x = tid%Lxsize;
        int y = tid/Lxsize;
        tx += devLattice[tid].type * cosf(k_x * x + k_y * y);
        ty += devLattice[tid].type * sinf(k_x * x + k_y * y);
    }

    rho_tilde_Re[index] = tx;
    rho_tilde_Im[index] = ty;
}