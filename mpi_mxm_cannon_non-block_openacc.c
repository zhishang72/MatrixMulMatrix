// Author: Zhi Shang <zhishang72@gmail.com>

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <math.h>
#ifdef _OPENACC
#include <openacc.h>
#endif /*_OPENACC*/
#include <mpi.h>

#define N 1024*4

int main(int argc, char *argv[])
{
    MPI_Comm cannon_comm;
    MPI_Status status;
    MPI_Request req1,req2,req3,req4;
    int rank,size;
    int shift,i,j,k;
    int dims[2], periods[2];
    int left,right,up,down;
    double *A,*B,*C;
    double *buf,*tmp=NULL;
    double start,tloc,tmax;
    unsigned int iseed=0;

    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    srand(iseed);

    dims[0]=0;    dims[1]=0;
    periods[0]=1; periods[1]=1;
    MPI_Dims_create(size,2,dims);
    if (dims[0]!=dims[1])
    {
        if(rank==0) printf("The number of processors must be a square.\n");
        MPI_Finalize();
        return 0;
    }

    int Nl=N/dims[0];
    A=(double*)malloc(Nl*Nl*sizeof(double));
    B=(double*)malloc(Nl*Nl*sizeof(double));
    C=(double*)calloc(Nl*Nl,sizeof(double));
    buf=(double*)malloc(Nl*Nl*sizeof(double));
    
    if(rank==0) printf("MPI processors: %4i\n",size);

    MPI_Cart_create(MPI_COMM_WORLD,2,dims,periods,1,&cannon_comm);
    MPI_Cart_shift(cannon_comm,0,1,&left,&right);
    MPI_Cart_shift(cannon_comm,1,1,&up,&down);
    
    bool accl = false;
    #if _OPENACC
        int ngpus=acc_get_num_devices(acc_device_nvidia);
        if (ngpus > 0) accl = true;
        int devicenum=rank%ngpus;
        printf("MPI rank: %4i; GPU device: %4i\n",rank,devicenum);
        acc_set_device_num(devicenum,acc_device_nvidia);
        // Call acc_init after acc_set_device_num to avoid 
        // multiple contexts on device 0 in multi GPU systems
        acc_init(acc_device_nvidia);
    #endif /*_OPENACC*/

    start=MPI_Wtime();
    for (i=0; i<Nl; i++)
    {
        for (j=0; j<Nl; j++)
        {
            A[i*Nl+j]=5-(int)( 10.0 * rand() / ( RAND_MAX + 1.0 ) );
            B[i*Nl+j]=5-(int)( 10.0 * rand() / ( RAND_MAX + 1.0 ) );
            C[i*Nl+j]=0.0;
        }
    }
   
    #pragma acc data copyin(A[Nl*Nl],B[Nl*Nl]), create(C[Nl*Nl]) async(rank) if(accl)     
    for (shift=0; shift<dims[0]; shift++)
    {
    // Matrix multiplication
        #pragma acc kernels async(rank) if(accl)
        {
            #pragma acc loop independent gang worker vector collapse(2)
            for (i=0; i<Nl; i++)    
                for (j=0; j<Nl; j++)    
                    C[i*Nl+j]=0.0;
            #pragma omp parallel for shared(Nl, A, B, C)
            #pragma acc loop independent gang worker vector collapse(3)
            for (i=0; i<Nl; i++)
                for (j=0; j<Nl; j++) 
                    for (k=0; k<Nl; k++)
                        C[i*Nl+k]+=A[i*Nl+j]*B[j*Nl+k];
        }
        #pragma acc update self(C)
   // Communication left - right
        MPI_Irecv(buf, Nl*Nl, MPI_DOUBLE, left, 100, cannon_comm, &req1);
        MPI_Isend(A, Nl*Nl, MPI_DOUBLE, right, 100, cannon_comm, &req2);
        tmp=buf; buf=A; A=tmp; 
        
   // Communication up - down
        MPI_Irecv(buf, Nl*Nl, MPI_DOUBLE, up, 200, cannon_comm, &req3);
        MPI_Isend(B, Nl*Nl, MPI_DOUBLE, down, 200, cannon_comm, &req4);
        tmp=buf; buf=B; B=tmp;
    }
    #pragma acc wait(rank)
    MPI_Wait(&req1, &status);
    MPI_Wait(&req2, &status);   
    MPI_Wait(&req3, &status);
    MPI_Wait(&req4, &status);
    tloc = MPI_Wtime() - start;
        
    MPI_Reduce(&tloc, &tmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if(rank==0) printf("CPU time: %.4fs\n",tmax);
    free(A); free(B); free(C); free(buf); 
    MPI_Finalize();
    return 0;
}


