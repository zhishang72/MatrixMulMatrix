// Author: Zhi Shang <zhishang72@gmail.com>

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <math.h>
#include <mpi.h>

#define N 1024*4

#define BLOCK_SIZE 32
//function without shared memory at GPU
__global__ 
void cuda_mxm_noshare(double *a, double *b, double *c, int n)
{ 
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if( col < n && row < n) 
    {
        c[row * n + col] = 0;
        for(int i = 0; i < n; i++) 
        {
            c[row * n + col] += a[row * n + i] * b[i * n + col];
        }
    }
}
//function with shared memory at GPU
__global__ 
void cuda_mxm_share(double *a, double *b, double *c, int n)
{
    __shared__ double sub_a[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ double sub_b[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int idx_a, idx_b, c_val = 0;

    for (int x = 0; x < gridDim.x; x++)
    {
        idx_a = row * n + x * BLOCK_SIZE + threadIdx.x;
        sub_a[threadIdx.y][threadIdx.x] = 0.0;
        if(idx_a < n*n)
        {
            // n may not be fully divided by BLOCK_SIZE
            sub_a[threadIdx.y][threadIdx.x] = a[idx_a];
        }
        idx_b = (x * BLOCK_SIZE + threadIdx.y) * n + col;
        sub_b[threadIdx.y][threadIdx.x] = 0.0;
        if(idx_a < n*n && idx_b < n*n)
        {
            // n may not be fully divided by BLOCK_SIZE
            sub_b[threadIdx.y][threadIdx.x] = b[idx_b];
        }
        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; k++)
        {
            c_val += sub_a[threadIdx.y][k] * sub_b[k][threadIdx.x];
        }
        __syncthreads();
    }
    if(row < n && col < n)
    {
        c[row * n + col] = c_val;
    }
}

int main(int argc, char *argv[])
{
    MPI_Comm cannon_comm;
    MPI_Status status;
    MPI_Request req1,req2,req3,req4;
    int rank,size;
    int shift,i,j;
    int dims[2], periods[2];
    int left,right,up,down;
    double *buf=NULL,*tmp=NULL;
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
    buf=(double*)malloc(Nl*Nl*sizeof(double));
    
    double *A=NULL,*B=NULL,*C=NULL;    
    cudaMallocManaged((void **) &A, sizeof(double)*Nl*Nl);
    cudaMallocManaged((void **) &B, sizeof(double)*Nl*Nl);
    cudaMallocManaged((void **) &C, sizeof(double)*Nl*Nl);

    unsigned int grid_rows = (Nl + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (Nl + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    
    if(rank==0) printf("MPI processors: %4i\n",size);

    MPI_Cart_create(MPI_COMM_WORLD,2,dims,periods,1,&cannon_comm);
    MPI_Cart_shift(cannon_comm,0,1,&left,&right);
    MPI_Cart_shift(cannon_comm,1,1,&up,&down);

    start=MPI_Wtime();
    for (i=0; i<Nl; i++)
    {
        for (j=0; j<Nl; j++)
        {
            A[i*Nl+j]=5-(int)( 10.0 * rand() / ( RAND_MAX + 1.0 ) );
            B[i*Nl+j]=5-(int)( 10.0 * rand() / ( RAND_MAX + 1.0 ) );
        }
    }
       
    for (shift=0; shift<dims[0]; shift++)
    {
    // Matrix multiplication
        //cuda_mxm_share<<<dimGrid, dimBlock>>>(A, B, C, Nl);     
        cuda_mxm_noshare<<<dimGrid, dimBlock>>>(A, B, C, Nl);
        // Transefr results from device to host 
        cudaDeviceSynchronize();
   // Communication left - right
        MPI_Irecv(buf, Nl*Nl, MPI_DOUBLE, left, 100, cannon_comm, &req1);
        MPI_Isend(A, Nl*Nl, MPI_DOUBLE, right, 100, cannon_comm, &req2);
        tmp=buf; buf=A; A=tmp; 
        
   // Communication up - down
        MPI_Irecv(buf, Nl*Nl, MPI_DOUBLE, up, 200, cannon_comm, &req3);
        MPI_Isend(B, Nl*Nl, MPI_DOUBLE, down, 200, cannon_comm, &req4);
        tmp=buf; buf=B; B=tmp;
    }
    MPI_Wait(&req1, &status);
    MPI_Wait(&req2, &status);   
    MPI_Wait(&req3, &status);
    MPI_Wait(&req4, &status);
    tloc = MPI_Wtime() - start;
        
    MPI_Reduce(&tloc, &tmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if(rank==0) printf("CPU time: %.4fs\n",tmax);
    cudaFree(A); cudaFree(B); cudaFree(C); free(buf);
    MPI_Finalize();
    return 0;
}


