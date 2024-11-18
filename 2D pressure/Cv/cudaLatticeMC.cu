#define R0_r (0.7)
#define MaxThreads (1024)

#include "subLatticeMC.c" 
#include <sys/stat.h>
#include <string.h>
#include <time.h>

void save_current_state(struct ptcl *hostPtcl, const int ptclNum, char *filename);
void read_state(struct ptcl *hostPtcl, int ptclNum, const char *filename);

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

     
        // Lattice and particle setting
        const double  Lx_size  = atof(argv[3]); //real_length
        const double  Ly_size  = atof(argv[4]); //real_length
        const double  Tmin     = atof(argv[5]); //real starting time
        const double  Tmax     = atof(argv[6]); //real ending time

        // Dynamics setting
        const int     n      = atoi(argv[7]);  
        const double  dt     = 0.1/pow(4,n);
        const double  dl     = 0.25/pow(2,n);
        const double  rho    = atof(argv[8]);
        const double  speed  = 1.0;

        const double  Dr     = 1.0;  // rotational diffusion
        const double  Dt     = 0.01; // translational diffusion coefficient

        const int tmin = (int)(Tmin/dt);
        const int tmax = (int)(Tmax/dt);    // real_time/dt

        // time measurement for initialization
        clock_t start, end;
        // time measurement for total simulation
        clock_t start1, end1;
        //double res;

        start = clock();
        
        srand(time(NULL));


        // total number of particles
        const  int   ptclNum = (int)(Lx_size * Ly_size * rho) ;
        
        const int nThreads = (MaxThreads<ptclNum)? MaxThreads : ptclNum;
        const int nBlocks  = (ptclNum+nThreads-1)/nThreads; 

        printf("%d %d\n", nThreads, nBlocks);
        
        size_t memSize = sizeof(struct ptcl) * ptclNum;
        
        struct ptcl *initPtcl;
        struct ptcl *hostPtcl;
        
        initPtcl = (struct ptcl *)malloc(sizeof(struct ptcl) * ptclNum);
        hostPtcl = (struct ptcl *)malloc(sizeof(struct ptcl) * ptclNum);
        
        double *devVangle;
        cudaMalloc(&devVangle, sizeof(double)*ptclNum);

        
        std::random_device rd;
        unsigned int seed = rd();
        // seed = 1234;
        // initialize the PRNGs
        curandState *devStates ;
        cudaMalloc((void **)&devStates, ptclNum*sizeof(curandState)) ;
        initialize_prng<<<nBlocks, nThreads>>>(ptclNum, seed, devStates) ;

        double free_Lx = Lx_size - 40.0;

        if(Tmin==0.0){
            //put free_Lx instead of L_size for proper pressure measure
            init_random_config_host(ptclNum, free_Lx, Ly_size, dl, initPtcl); 
        }
        else{
            // If Tmin!=0 load from saved state.
            char state_filename[200];
            sprintf(state_filename, "%s/state_t_%d.txt", folder_name, tmin);
            read_state(initPtcl, ptclNum, state_filename);
        }
        
        printf("total particle number = %d\n", ptclNum);
        
        struct ptcl *devPtcl ;
        // particles in the device
        cudaMalloc(&devPtcl, sizeof(struct ptcl)*ptclNum) ;
        
        cudaMemcpy(devPtcl, initPtcl, memSize, cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        
        end = clock();
        printf("intialize (in sec) : %f\n", (double)(end - start) / CLOCKS_PER_SEC);
        
        
        // reptition start  
        
        start1 = clock();
        
        for (int t = tmin; t < tmax + 1; t++){
        
            move<<<nBlocks, nThreads>>>(devStates, Lx_size, Ly_size, ptclNum, speed, dt, dl, Dt, Dr, devPtcl);
                
            cudaDeviceSynchronize();

            //cudaMemcpy(hostPtcl, devPtcl, memSize, cudaMemcpyDeviceToHost);
            //cudaMemcpy(devPtcl, hostPtcl, memSize, cudaMemcpyHostToDevice);
            
            // set save_interval here
            if (t%((int)(100/dt))==0 && t>= 0){
                printf("t_step=%d\n", t);
                cudaMemcpy(hostPtcl, devPtcl, memSize, cudaMemcpyDeviceToHost);
                char state_filename[200];
                sprintf(state_filename, "%s/state_t_%d.txt", folder_name, t);
                save_current_state(hostPtcl, ptclNum, state_filename);
                
                double real_t = (double)t * dt;
                printf("T=%d\n", (int)real_t);
            }       
        }
        
        end1 = clock();
        printf("Total sim time (in sec) : %f\n", (double)(end1 - start1) / CLOCKS_PER_SEC);
        
        
        free(hostPtcl); free(initPtcl); 
        
        cudaFree(devPtcl); cudaFree(devVangle); cudaFree(devStates);
}



void save_current_state(struct ptcl *hostPtcl, const int ptclNum, char *filename){
        FILE *fp = fopen(filename, "w");
        if (fp == NULL) {printf("Error opening the file %s", filename); return;}
        
        for (int i = 0; i < ptclNum; i++) {
                fprintf(fp, "%d %d %lf %lf ", hostPtcl[i].x, hostPtcl[i].y, hostPtcl[i].theta, hostPtcl[i].force);
        }
        fprintf(fp, "\n");
        fclose(fp);
        
}

void read_state(struct ptcl *hostPtcl, int ptclNum, const char *filename) {
    FILE *fp = fopen(filename, "r");
    if (fp == NULL) {
        printf("Error opening the file %s\n", filename);
        return;
    }

    for (int i = 0; i < ptclNum; i++) {
        if (fscanf(fp, "%d %d %lf %lf", &hostPtcl[i].x, &hostPtcl[i].y, &hostPtcl[i].theta, &hostPtcl[i].force) != 4) {
            printf("Error reading data at entry %d\n", i);
            break;
        }
    }

    fclose(fp);
}
