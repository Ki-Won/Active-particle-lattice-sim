#define R0_r (0.7)
#define MaxThreads (1024)

#include "subLatticeMC.c" 
#include <sys/stat.h>
#include <string.h>
#include <time.h>

void save_current_state(struct ptcl *hostPtcl, const int ptclNum, char *filename, const double T);
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
        const double  L_size  = atoi(argv[3]);  // real_length
        const double    Tmin  = atoi(argv[4]);  // real starting time
        const double    Tmax  = atoi(argv[5]);  // real ending time

        // Dynamics setting
        const int     n      = atoi(argv[6]);   
        const double  dt     = 0.1/pow(4,n);
        const double  dl     = 0.25/pow(2,n);
        const double  rho    = atof(argv[7]);
        const double  speed  = 1.0;

        const double  alpha  = 0.02; // tumbling rate
        const double  Dt     = 0.013; // translational diffusion coefficient
        
        const int     type   = atoi(argv[8]); // Cv or C0pot
        if(type < 0 || type > 1) error_output("invalid type") ;


        const long tmin = (long)(Tmin/dt);
        const long tmax = (long)(Tmax/dt);    // real_time/dt

        // time measurement for initialization
        clock_t start, end;
        // time measurement for total simulation
        clock_t start1, end1;
        //double res;

        start = clock();
        
        srand(time(NULL));

        // total number of particles
        const  int   ptclNum = (int)((L_size)*rho) ; 
        
        // grid dimension
        int sys_size = (int)(L_size/dl);
        
        const int nThreads = (MaxThreads<ptclNum)? MaxThreads : ptclNum;
        const int nBlocks  = (ptclNum+nThreads-1)/nThreads; 
        
        const int nThreads_l = (MaxThreads<sys_size)? MaxThreads : sys_size;
        const int nBlocks_l  = (sys_size+nThreads_l-1)/nThreads_l; 

        printf("%d %d\n", nThreads, nBlocks);
        
        size_t memSize = sizeof(struct ptcl) * ptclNum;
        
        struct ptcl *initPtcl;
        struct ptcl *hostPtcl;
        
        initPtcl = (struct ptcl *)malloc(sizeof(struct ptcl) * ptclNum);
        hostPtcl = (struct ptcl *)malloc(sizeof(struct ptcl) * ptclNum);
        
        double *devVangle;
        cudaMalloc(&devVangle, sizeof(double)*ptclNum);
        
        // Initialize arrays for force, potential, and transition rates
        double *F; double *VL; double *VC; double *VR;
        cudaMalloc(&F, sizeof(double) * sys_size);
        cudaMalloc(&VL, sizeof(double) * sys_size);
        cudaMalloc(&VC, sizeof(double) * sys_size);
        cudaMalloc(&VR, sizeof(double) * sys_size);

        double *host_F; double *host_VL; double *host_VC; double *host_VR;
        host_F = (double *)malloc(sizeof(double) * sys_size);
        host_VL = (double *)malloc(sizeof(double) * sys_size);
        host_VC = (double *)malloc(sizeof(double) * sys_size);
        host_VR = (double *)malloc(sizeof(double) * sys_size);

        double *Wlp; double *Wlm; double *Wrp; double *Wrm;
        cudaMalloc(&Wlp, sizeof(double) * sys_size);
        cudaMalloc(&Wlm, sizeof(double) * sys_size);
        cudaMalloc(&Wrp, sizeof(double) * sys_size);
        cudaMalloc(&Wrm, sizeof(double) * sys_size);

        double *host_Wlp; double *host_Wlm; double *host_Wrp; double *host_Wrm;
        host_Wlp = (double *)malloc(sizeof(double) * sys_size);
        host_Wlm = (double *)malloc(sizeof(double) * sys_size);
        host_Wrp = (double *)malloc(sizeof(double) * sys_size);
        host_Wrm = (double *)malloc(sizeof(double) * sys_size);
        
        init_FV<<<nBlocks_l, nThreads_l>>>(F, VL, VC, VR, sys_size, L_size, dl, Dt, speed);
        if (type == 0) {init_W_Cv<<<nBlocks_l, nThreads_l>>>(VL, VC, VR, Wlp, Wlm, Wrp, Wrm, sys_size, dt, dl, Dt, speed);}
        if (type == 1) {init_W_C0pot<<<nBlocks_l, nThreads_l>>>(VL, VC, VR, Wlp, Wlm, Wrp, Wrm, sys_size, dt, dl, Dt, speed);}


        cudaMemcpy(host_F, F, sizeof(double) * sys_size, cudaMemcpyDeviceToHost);
        cudaMemcpy(host_VL, VL, sizeof(double) * sys_size, cudaMemcpyDeviceToHost);
        cudaMemcpy(host_VC, VC, sizeof(double) * sys_size, cudaMemcpyDeviceToHost);
        cudaMemcpy(host_VR, VR, sizeof(double) * sys_size, cudaMemcpyDeviceToHost);

        // Output force and potential to txt file
        char Fname[200];
        sprintf(Fname, "%s/FV.txt", folder_name);
        FILE *fp = fopen(Fname, "w");
        for (int i = 0; i < sys_size; i++){
            fprintf(fp, "%f %f %f %f ", host_F[i], host_VL[i], host_VC[i], host_VR[i]);
        }
        fclose(fp);

        cudaMemcpy(host_Wlp, Wlp, sizeof(double) * sys_size, cudaMemcpyDeviceToHost);
        cudaMemcpy(host_Wrp, Wrp, sizeof(double) * sys_size, cudaMemcpyDeviceToHost);
        cudaMemcpy(host_Wlm, Wlm, sizeof(double) * sys_size, cudaMemcpyDeviceToHost);
        cudaMemcpy(host_Wrm, Wrm, sizeof(double) * sys_size, cudaMemcpyDeviceToHost);
        
        // Output transition rates to txt file
        char Wname[200];
        sprintf(Wname, "%s/W.txt", folder_name);
        fp = fopen(Wname, "w");
        for (int i = 0; i < sys_size; i++){
            fprintf(fp, "%f %f %f %f ", host_Wlp[i], host_Wlm[i], host_Wrp[i], host_Wrm[i]);
        }
        fclose(fp);       
        
     
        std::random_device rd;
        unsigned int seed = rd();
        // seed = 1234;
        // initialize the PRNGs
        curandState *devStates ;
        cudaMalloc((void **)&devStates, ptclNum*sizeof(curandState)) ;
        initialize_prng<<<nBlocks, nThreads>>>(ptclNum, seed, devStates) ;

        if(Tmin==0.0){
            init_random_config_host(ptclNum, L_size, dl, initPtcl); 
        }
        else{
            // If Tmin!=0 load from saved state.
            char state_filename[200];
            sprintf(state_filename, "%s/state_t_%ld.txt", folder_name, tmin);
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

        printf("tmax = %ld\n", tmax);
        double step;
        step = 0.0;
        
        for (long t = tmin; t < tmax + 1; t++){
        
            move<<<nBlocks, nThreads>>>(devStates, L_size, ptclNum, speed, dt, dl, Dt, alpha, VL, VC, VR, Wlp, Wlm, Wrp, Wrm, devPtcl);
            step += 1.0;
                
            cudaDeviceSynchronize();

            //cudaMemcpy(hostPtcl, devPtcl, memSize, cudaMemcpyDeviceToHost);
            //cudaMemcpy(devPtcl, hostPtcl, memSize, cudaMemcpyHostToDevice);
            
            // set save_interval here
            if (t%((long)(1000/dt))==0 && t>= 0 && t<((long)(30000/dt))){
                printf("t_step=%ld\n", t);
                cudaMemcpy(hostPtcl, devPtcl, memSize, cudaMemcpyDeviceToHost);
                reset<<<nBlocks, nThreads>>>(devPtcl);
                char state_filename[200];
                sprintf(state_filename, "%s/state_t_%ld.txt", folder_name, t);
                save_current_state(hostPtcl, ptclNum, state_filename, step*dt);
                step = 0.0;

                double real_t = (double)t * dt;

                printf("T=%d\n", (int)real_t);
            }

            // set save_interval2 here
            if (t%((long)(100/dt))==0 && t>= ((long)(30000/dt))){
                cudaMemcpy(hostPtcl, devPtcl, memSize, cudaMemcpyDeviceToHost);
                reset<<<nBlocks, nThreads>>>(devPtcl);
                char state_filename[200];
                sprintf(state_filename, "%s/state_t_%ld.txt", folder_name, t);
                save_current_state(hostPtcl, ptclNum, state_filename, step*dt);
                step = 0.0;

                double real_t = (double)t * dt;
                printf("T=%d\n", (int)real_t);
            }
                
        }
        
        end1 = clock();
        printf("Total sim time (in sec) : %f\n", (double)(end1 - start1) / CLOCKS_PER_SEC);
        
        
        free(hostPtcl); free(initPtcl); 
        
        cudaFree(devPtcl); cudaFree(devVangle); cudaFree(devStates);
        
        free(host_F); free(host_VL); free(host_VC); free(host_VR); free(host_Wlp); free(host_Wlm); free(host_Wrp); free(host_Wrm);
        
        cudaFree(F); cudaFree(VL); cudaFree(VC); cudaFree(VR); free(Wlp); free(Wlm); free(Wrp); free(Wrm);
}



void save_current_state(struct ptcl *hostPtcl, const int ptclNum, char *filename, const double T){
        FILE *fp = fopen(filename, "w");
        if (fp == NULL) {printf("Error opening the file %s", filename); return;}
        //printf("step = %f\n", T);
        for (int i = 0; i < ptclNum; i++) {
                fprintf(fp, "%d %d %lf %lf ", hostPtcl[i].x, hostPtcl[i].s, hostPtcl[i].even_EP/T, hostPtcl[i].J/T);
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
        if (fscanf(fp, "%d %d %lf %lf", &hostPtcl[i].x, &hostPtcl[i].s, &hostPtcl[i].even_EP, &hostPtcl[i].J) != 4) {
            printf("Error reading data at entry %d\n", i);
            break;
        }
    }

    fclose(fp);
}
