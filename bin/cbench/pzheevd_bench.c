#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "mpi.h"
#include <math.h>

#include <sys/time.h>

#include <complex.h>

#ifdef _OPENMP
#include <omp.h>
#endif

typedef double complex complex16;

int main(int argc, char **argv) {

  int ictxt, nside, ngrid, nblock, nthread;
  int rank, size;
  int ic, ir, nc, nr;

  int i, j;

  char *fname;
  
  int info, ZERO=0, ONE=1;

  struct timeval st, et;

  double dtev;
  double gfpc_ev;

  /* Initialising MPI stuff */
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  printf("Process %i of %i.\n", rank, size);


  /* Parsing arguments */
  if(argc < 6) {
    exit(-3);
  }

  nside = atoi(argv[1]);
  ngrid = atoi(argv[2]);
  nblock = atoi(argv[3]);
  nthread = atoi(argv[4]);
  fname = argv[5];

  if(rank == 0) {
    printf("Multiplying matrices of size %i x %i\n", nside, nside);
    printf("Process grid size %i x %i\n", ngrid, ngrid);
    printf("Block size %i x %i\n", nblock, nblock);
    printf("Using %i OpenMP threads\n", nthread);
  }



  #ifdef _OPENMP
  if(rank == 0) printf("Setting OMP_NUM_THREADS=%i\n", nthread);
  omp_set_num_threads(nthread);
  #endif

  /* Setting up BLACS */
  Cblacs_pinfo( &rank, &size ) ;
  Cblacs_get(-1, 0, &ictxt );
  Cblacs_gridinit(&ictxt, "Row", ngrid, ngrid);
  Cblacs_gridinfo(ictxt, &nr, &nc, &ir, &ic);

  int descA[9], descev[9];

  /* Fetch local array sizes */
  int Ar, Ac;

  Ar = numroc_( &nside, &nblock, &ir, &ZERO, &nr);
  Ac = numroc_( &nside, &nblock, &ic, &ZERO, &nc);

  printf("Local array section %i x %i\n", Ar, Ac);

  /* Set descriptors */
  descinit_(descA, &nside, &nside, &nblock, &nblock, &ZERO, &ZERO, &ictxt, &Ar, &info);
  descinit_(descev, &nside, &nside, &nblock, &nblock, &ZERO, &ZERO, &ictxt, &Ar, &info);  

  /* Initialise and fill arrays */
  complex16 *A = (complex16 *)malloc(Ar*Ac*sizeof(complex16));
  complex16 *evecs = (complex16 *)malloc(Ar*Ac*sizeof(complex16));

  double *evals = (double *)malloc(nside * sizeof(double));

  for(i = 0; i < Ar; i++) {
    for(j = 0; j < Ac; j++) {
      A[j*Ar + i] = drand48();
      evecs[j*Ar + i] = 0.0;
    }
  }

  int liwork = -1;
  int tli = 0;


  char * uplo = "U";
  

  // Workspace size inquiry
  int lwork = -1;
  complex16 tlw = 0.0;

  int lrwork = -1;
  double tlr = 0.0;

  pzheevd_("V", "U", &nside,
             A, &ONE, &ONE, descA,
             evals,
             evecs, &ONE, &ONE, descev,
             &tlw, &lwork,
             &tlr, &lrwork,
             &tli, &liwork,
             &info);
    
  // Initialise workspace to correct length
  lwork = (int)tlw;
  complex16 * work = (complex16 *)malloc(sizeof(complex16) * lwork);

  lrwork = (int)tlr;
  double * rwork = (double *)malloc(sizeof(double) * lrwork);

  liwork = (int)tli;
  int * iwork = (int *)malloc(sizeof(int) * liwork);


  if(rank == 0) printf("Starting eigenvalue.\n");

  Cblacs_barrier(ictxt,"A");
  gettimeofday(&st, NULL);

  // Compute eigen problem
  pzheevd_("V", "U", &nside,
           A, &ONE, &ONE, descA,
           evals,
           evecs, &ONE, &ONE, descev,
           work, &lwork,
           rwork, &lrwork,
           iwork, &liwork,
           &info);


  Cblacs_barrier(ictxt,"A");
  gettimeofday(&et, NULL);
  dtev = (double)((et.tv_sec-st.tv_sec) + (et.tv_usec-st.tv_usec)*1e-6);
  gfpc_ev = 16.0*pow(nside, 3) / (dtev * 1e9 * ngrid * ngrid * nthread);


  if(rank == 0) printf("Done.\n=========\nTime taken: %g s\nGFlops per core: %g\n=========\n", dtev, gfpc_ev);


  free(iwork);
  free(work);
  free(A);
  free(evecs);


  if(rank == 0) {
    FILE * fd;
    fd = fopen(fname, "w");
    fprintf(fd, "%g %g %g %g %i %i %i %i %g %g %g %g\n", gfpc_ev, nside, ngrid, nblock, nthread, dtev);
    fclose(fd);
  }

  Cblacs_gridexit( 0 );
  MPI_Finalize();

}

