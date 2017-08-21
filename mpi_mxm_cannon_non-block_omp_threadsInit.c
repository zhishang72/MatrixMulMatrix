// Author: Zhi Shang <zhishang72@gmail.com>

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>

#define N 1024*4

int main(int argc, char *argv[])
{
    MPI_Comm cannon_comm;
    MPI_Status status;
    MPI_Request req1,req2,req3,req4;
    int rank,size,threads,provided;
    int shift,i,j,k;
    int dims[2], periods[2];
    int left,right,up,down;
    double *A,*B,*C;
    double *buf,*tmp=NULL;
    double start,tloc,tmax;
    unsigned int iseed=0;
    
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
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
    
    #pragma omp parallel
    {
        threads = omp_get_num_threads();
    }
    if(rank==0) printf("MPI processors: %4i; OpenMP threads: %4i\n",size,threads);

    MPI_Cart_create(MPI_COMM_WORLD,2,dims,periods,1,&cannon_comm);
    MPI_Cart_shift(cannon_comm,0,1,&left,&right);
    MPI_Cart_shift(cannon_comm,1,1,&up,&down);

    start=MPI_Wtime();
    #pragma omp parallel private(i,j)
    //#pragma omp for schedule(static)  //OpenMP-2
    #pragma omp for schedule(static) collapse(2)//OpenMP-3
    //#pragma omp for simd collapse(2) aligned(A,B,C:64) safelen(32) //OpenMP-4
    for (i=0; i<Nl; i++)
    {
        for (j=0; j<Nl; j++)
        {
            A[i*Nl+j]=5-(int)( 10.0 * rand() / ( RAND_MAX + 1.0 ) );
            B[i*Nl+j]=5-(int)( 10.0 * rand() / ( RAND_MAX + 1.0 ) );
            C[i*Nl+j]=0.0;
        }
    }
    
    for (shift=0; shift<dims[0]; shift++)
    {
    // Matrix multiplication
        #pragma omp parallel 
        {
           //#pragma omp for schedule(static) private(i,j,k) //OpenMP-2
           #pragma omp for schedule(static) collapse(3) private(i,j,k) //OpenMP-3
           //#pragma omp for simd collapse(3) aligned(A,B,C:64) safelen(32) private(i,j,k) //OpenMP-4
           for (i=0; i<Nl; i++) for (j=0; j<Nl; j++) for (k=0; k<Nl; k++) C[i*Nl+k]+=A[i*Nl+j]*B[j*Nl+k];
           
           #pragma omp single //OpenMP Multiple
           {
           // Communication left - right
               MPI_Irecv(buf, Nl*Nl, MPI_DOUBLE, left, 100, cannon_comm, &req1);
               MPI_Isend(A, Nl*Nl, MPI_DOUBLE, right, 100, cannon_comm, &req2);
               tmp=buf; buf=A; A=tmp;
        
           // Communication up - down
               MPI_Irecv(buf, Nl*Nl, MPI_DOUBLE, up, 200, cannon_comm, &req3);
               MPI_Isend(B, Nl*Nl, MPI_DOUBLE, down, 200, cannon_comm, &req4);
               tmp=buf; buf=B; B=tmp;
           }
        }
    }
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


