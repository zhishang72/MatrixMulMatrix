#
# Author: Zhi Shang
#
CC	= mpicc 
#
#SRCS    = mpi_mxm_cannon_standard.c
#SRCS    = mpi_mxm_cannon_non-block.c
SRCS    = mpi_mxm_cannon_syncro-block.c
#
#OBJECTS = mpi_mxm_cannon_standard.o 
#OBJECTS = mpi_mxm_cannon_non-block.o
OBJECTS = mpi_mxm_cannon_syncro-block.o
#
LIBES	= 
#
CFLAGS	= -O3 

mpi_mxm_cannon:	$(OBJECTS)
		$(CC) $(CFLAGS) -o mpi_mxm_cannon $(OBJECTS) $(LIBES)

clean:
	rm -f *.o mpi_mxm_cannon *~ *.dat
