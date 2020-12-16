// parallel tiled Cholesky solver test program
// written by Peter Strazdins, Feb 20 for COMP4300/8300 Assignment 1
// v1.0 27 Feb 

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h> //getopt()
#include <math.h>   //fabs()
#include <assert.h>
#include <mpi.h>
#ifdef USE_MKL
#include <mkl_cblas.h>
#else
#include <cblas.h>
#endif   
#include "auxCholesky.h"
#include "checkCholesky.h"
#include "parCholesky.h"

#define USAGE   "testCholesky [-r|-b] [-s s] [-p P] [-e|-x] [-v v] [-t t] N [wT]"
#define DEFAULTS "P=1 s=v=t=0 wT=N"
#define CONSTRAINTS "1<=P<=nprocs, N>=0, 1<=wT<=N"
#define OPTCHARS "rbp:exs:v:t:"

static int N;                  // matrix size
static int wT = 0;             // tile width
static int randDist = 0;       // use a random distribution; set if -r given
static int seed = 0;           // s, above; random number generator seed
static int blockDist = 0;      // use a block distribution; set if -b given
                               // use a cyclic distribution, if neither -r, -b
static int P=1, Q;             // use a PxQ logical process grid, Q = nprocs/P
                               // (not applicable for the random distribution)
static int eagerComm = 0;      // use eager comm. algorithm; set if -e given
static int optX = 0;           // use `extra' algorithm; set if -x given
static int tuneParam = 0;      // t, above; optional tuning parameter
static int verbosity = 0;      // v, above; output verbosity parameter
static int id, nprocs;         // MPI parameters

// print a usage message for this program and exit with a status of 1
void usage(char *msg) {
  if (id==0) {
    printf("testCholesky: %s\n", msg);
    printf("\tusage: %s\n\tdefault values: %s\n", USAGE, DEFAULTS);
    fflush(stdout);
  }
  exit(2);
}

void getArgs(int argc, char *argv[]) {
  extern char *optarg; // points to option argument (for -p option)
  extern int optind;   // index of last option parsed by getopt()
  extern int opterr;
  char optchar;        // option character returned my getopt()
  opterr = 0;          // suppress getopt() error message for invalid option
  while ((optchar = getopt(argc, argv, OPTCHARS)) != -1) {
    // extract next option from the command line     
    switch (optchar) {
    case 'r':
      randDist = 1;
      break;
    case 'b':
      blockDist = 1;
      break;
    case 'p':
      if (sscanf(optarg, "%d", &P) != 1) // invalid integer 
	usage("bad value for P");
      break;
    case 's':
      if (sscanf(optarg, "%d", &seed) != 1) // invalid integer 
	usage("bad value for s");
      break;
    case 'v':
      if (sscanf(optarg, "%d", &verbosity) != 1) // invalid integer 
	usage("bad value for v");
      break;
    case 't':
      if (sscanf(optarg, "%d", &tuneParam) != 1) // invalid integer 
	usage("bad value for t");
      break;
    case 'e':
      eagerComm = 1;
      break;
    case 'x':
      optX = 1;
      break;
    default:
      usage("unknown option");
      break;
    } //switch 
   } //while

  if (!randDist && (P <= 0 || nprocs < P))
    usage("P must be in range 1..nprocs");
  Q = nprocs / P;

  if (optind < argc) {
    if (sscanf(argv[optind], "%d", &N) != 1)
      usage("bad value for N");
    if (N < 0)
      usage("N must be >= 0");
  } else
    usage("missing N");
  wT = N;
  if (optind+1 < argc) {
    if (sscanf(argv[optind+1], "%d", &wT) != 1) 
      usage("bad value for wT");
    if (wT < 1 || wT > N)
      usage("wT must be in range 1..N");
  }
} //getArgs()

void printTime(char * stage, double gflops, double t) {
  if (id == 0) {
    printf("%s time %.2es, GFLOPs rate=%.2e (per process %.2e)\n",
	   stage, t, gflops / t,  gflops / t / nprocs); 
  }
} //printTime()

#define MAX_RESID 1      // largest acceptable normalized residual
#define EPSILON 2.0E-16  // machine precision for double 

int main(int argc, char** argv) {
  double ***A; int nT; // A stores local tiles of global array's nTxnT tiles 
  int **ownerIdTile;         // ownerIdTile[i][j] give id of tile (i,j)
  double *x, *y;       // replicated Nx1 vectors used for checking
  double t,t1,t2;            // for recording time
  int i, j;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &id);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  getArgs(argc, argv);
  if (id == 0) {
    printf("Cholesky factor of a %dx%d matrix with %dx%d tiles distributed ",
	   N, N, wT, wT);
    if (randDist)
      printf("randomly over %d processes\n", nprocs);
    else
      printf("%s over %dx%d processes\n", 
	     blockDist? "blockwise": "cyclically", P, Q);
    if (optX)
      printf("\tUsing extra optimization method\n");
    else if (eagerComm)
      printf("\tUsing eager communication\n");
    if (seed != 0 || tuneParam != 0)
      printf("\tWith random seed %d, tuning parameter %d\n", seed, tuneParam);
  }
  initParams(randDist, seed, blockDist, P, Q, optX, verbosity, tuneParam);
  setPrintDoublePrecision(2); //print doubles to 2 decimal places
  
  nT = (N + wT-1) / wT;
  ownerIdTile = (int **) allocTileArray(sizeof(int), nT);
  initOwners(ownerIdTile, nT);
  if (id == 0 && verbosity > 0)
    printIntTile(id, nT, ownerIdTile);

  A = (double ***) allocTileArray(sizeof(double *), nT);
  MPI_Barrier(MPI_COMM_WORLD); t,t1 = MPI_Wtime();
  for (i = 0; i < nT; i++)
    for (j = 0; j <= i; j++) {
      assert(A[i][j] == NULL); //expected from allocTileArray()
      if (id == ownerIdTile[i][j]) {
	A[i][j] = (double *) malloc(wT*wT*sizeof(double));
	assert(A[i][j] != NULL);
      	initLowerPosDefTile(N, i, j, seed, wT, A[i][j]);
      }
    }
  MPI_Barrier(MPI_COMM_WORLD);
  if (verbosity > 0) //this exposed a severe perf. bug in initLowerPosDefTile()
    printTime("Generate matrix", 1.0e-09 * N * N, MPI_Wtime() - t);
  if (verbosity > 2)
    printLowerTileArray(id, nT, wT, A);

  x = (double *) malloc(nT*wT * sizeof(double));
  y = (double *) malloc(nT*wT * sizeof(double));
  assert (x != NULL && y != NULL);
  initVec(seed, x, nT*wT);
  if (verbosity > 1 && id==0)
    printVec("x", x, nT*wT);
  MPI_Barrier(MPI_COMM_WORLD); t = MPI_Wtime();
  triMatVecMult(nT, wT, ownerIdTile, A, x, y); // y = A*x 
  MPI_Barrier(MPI_COMM_WORLD); 
  printTime("Generate RHS", 1.0e-09 * 2.0 * N * N, MPI_Wtime() - t);
  if (verbosity > 1 && id==0)
    printVec("y", y, nT*wT);

  MPI_Barrier(MPI_COMM_WORLD); t = MPI_Wtime();
  if (optX)
    choleskyX(ownerIdTile, nT, wT, A);
  else if (eagerComm)    
    choleskyEager(ownerIdTile, nT, wT, A); 
  else
    choleskyLazy(ownerIdTile, nT, wT, A);
  MPI_Barrier(MPI_COMM_WORLD);
  printTime("Factorization", 1.0e-09 * 2.0/3.0 * N * N * N, MPI_Wtime() - t);
  if (verbosity > 1)
    printLowerTileArray(id, nT, wT, A);    

  MPI_Barrier(MPI_COMM_WORLD); t = MPI_Wtime();
  triMatVecSolve(nT, wT, ownerIdTile, A, y); //y = A^-1 * y; should now be ~= x
  MPI_Barrier(MPI_COMM_WORLD); 
  t2 = MPI_Wtime();
  printTime("Backsolve", 1.0e-09 * 2.0 * N * N, t2 - t);
  printTime("Total", 1.0e-09 * 2.0 * N * N, t2-t1);
  if (verbosity > 1 && id==0)
    printVec("x'", y, nT*wT);
  if (verbosity > 1 && id==0)
    printVec(" x", x, nT*wT);
  if (id == 0) { 
    double resid, normX = fabs(x[cblas_idamax(N, x, 1)]);     
    cblas_daxpy(N, -1.0, y, 1, x, 1); //x = x - y, should now be ~= 0
    resid = fabs(x[cblas_idamax(N, x, 1)]);
    resid /= (getNrmA(N) * normX * EPSILON);
    printf("%sed residual check: %1.2e\n",
          (resid > MAX_RESID || resid != resid /*true for +/-NaN*/)? 
	   "FAIL": "PASS", resid);
  }

  // free all data; note buffer overwrites may cause these to crash
  free(y); free(x);
  for (i = 0; i < nT; i++)
    for (j = 0; j < nT; j++) 
      if (A[i][j] != NULL) //note A[i][j] was possibly allocated in cholesky*()
	free(A[i][j]);
  freeTileArray((void**) A);
  freeTileArray((void**) ownerIdTile);

  MPI_Finalize();
  return 0;
} //main()

