#
# Author: Zhi Shang <zhishang72@gmail.com>
#
CC	= mpicc #for OpenACC the mpicc should be installed to link PGI pgcc
#
#SRCS    = mpi_mxm_cannon_standard.c
#SRCS    = mpi_mxm_cannon_non-block.c
#SRCS    = mpi_mxm_cannon_syncro-block.c
SRCS    = mpi_mxm_cannon_non-block_omp_threadsInit.c
#SRCS    = mpi_mxm_cannon_non-block_openacc.c
#
#OBJECTS = mpi_mxm_cannon_standard.o 
#OBJECTS = mpi_mxm_cannon_non-block.o
#OBJECTS = mpi_mxm_cannon_syncro-block.o
OBJECTS = mpi_mxm_cannon_non-block_omp_threadsInit.o
#OBJECTS = mpi_mxm_cannon_non-block_openacc.o
#
INCLUDE	= 
#
LIBES	= -lm -lmpi 
#gcc or mpicc for OpenMP
CFLAGS	= -O3 -Wall -fopenmp -fopenmp-simd -ftree-vectorize
#pgcc or mpicc for OpenACC
#Pgprof  = -Minfo=accel,ccff #-Mprof=func,lines,time 
#CFLAGS	= -O3 -acc $(Pgprof) -ta=tesla:managed #-ta=nvidia 
#
mxm_cannon:$(OBJECTS)
	$(CC) $(CFLAGS) $(INCLUDE) -o mxm_cannon $(OBJECTS) $(LIBES) 
#
clean:
	rm -f *.o mxm_cannon *~ *.dat
