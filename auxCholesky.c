// auxiliary (non-parallel) functions for supporting tiled Cholesky factor.
// written by Peter Strazdins, Feb 20 for COMP4300/8300 Assignment 1 
// v1.0 26/02/20

#include <stdio.h>
#include <stdlib.h>
#include <math.h> //fabs()
#include <assert.h>

#include "auxCholesky.h"

void **allocTileArray(size_t eltSize, int nT) {
  int i;
  char **A = (char **) malloc(nT*sizeof(void *));  
  A[0] = (char *) calloc(eltSize, nT*nT);
  for (i=1; i<nT; i++)
    A[i] = A[i-1] + nT*eltSize;
  return (void **) A;
} //allocTileArray()

void freeTileArray(void **A) {
  free(A[0]);
  free(A);
} //freeTileArray()


void printIntTile(int id, int nT, int **a) {
  int i, j;
  for (i=0; i < nT; i++) {
    printf("%d:", id);
    for (j=0; j < nT; j++) 
      printf(" %2d", a[i][j]); 
    printf("\n");
  }
} //printIntTile()

// printing out to 2 decimal places will detect most errors, but some will
// require greater precision before a wrong value becomes apparent
static char *doubleFmt = " %+5.2f";
static char *doubleSpFmt = "%6s"; // equivalent amount of spaces
static char doubleFmtBuf[32], doubleSpFmtBuf[32];

void setPrintDoublePrecision(int decimalPlaces) {
  sprintf(doubleFmtBuf, " %%+%d.%df", decimalPlaces+3, decimalPlaces);
  sprintf(doubleSpFmtBuf, " %%%ds", decimalPlaces+3);  
  doubleFmt = doubleFmtBuf, doubleSpFmt = doubleSpFmtBuf;
} //setDoublePrecision()

void printDoubleTile(int id, int wT, double *a) {
  int i, j;
  for (i=0; i < wT; i++) {
    printf("%d:", id);
    for (j=0; j < wT; j++) 
      printf(doubleFmt, a[i*wT+j]); 
    printf("\n");
  }
} //printDoubleTile()


// return a diagonal bias (minimally) sufficient to make a matrix with
// random elements in (-1,+1) positive definite
static double diagBias(int N) {
  return (1.0 + N / 4.0); // the scaling factor 4.0 is empirically determined
}


// seeding the random number generators for every element ensures
// the value of every element depends only on its global index 
// and the seed (i.e. does not depend on wT); this is useful for debugging
// the (serial) tiled algorithm via printing the matrices.
// However seeding is costly, so avoid this when matrix is too large to print
#define N_MAX_PRINT_THRESHOLD 50

void initLowerPosDefTile(int N, int i0, int j0, int seed, int wT, double *a) {
  int i, j;
  int seedOnceOnly = (N > N_MAX_PRINT_THRESHOLD);
  if (seedOnceOnly)
    srand(i0 + j0 + 29*seed);     
  for (i=0; i < wT; i++) 
    for (j=0; j < wT; j++) {
      int iG = i0*wT+i, jG = j0*wT+j;
      if (iG < jG) //upper triangular element 
	a[i*wT+j] = 0.0;
      else if (iG >= N) { //in a padded-out row from when N%wT > 0
	assert (iG < ((N+wT-1)/wT)*wT);
      	a[i*wT+j] = (double) (iG == jG); //pad out with the identity matrix
      } else {
	if (!seedOnceOnly)
	  srand(iG + jG + 29*seed); 
	a[i*wT+j] = (2.0 * rand() / RAND_MAX) - 1.0;
	assert (-1.0 <= a[i*wT+j] && a[i*wT+j] <= 1.0);
	if (iG == jG)
	  a[i*wT+j] = fabs(a[i*wT+j]) + diagBias(N);
      }
    } //for (j...)
} //initLowerPosDefTile()


double getNrmA(int N) {
  // approximate max row sum of a matrix with each row 
  // having N elements random in [-1,+1] and the
  // diagonal with a bias of diagBias(N) added to
  return (N/2 + diagBias(N));
} //getNrmA()


void printLowerTileArray(int id, int nT, int wT, double ***a) {
  int i0, j0, i, j;
  for (i0=0; i0 < nT; i0++) 
    for (i=0; i < wT; i++) {
      printf("%d:", id);
      for (j0=0; j0 < nT; j0++) 
	for (j=0; j < wT; j++)
	  if (a[i0][j0] == NULL)
	    printf(doubleSpFmt, " ");
	  else
	    printf(doubleFmt, a[i0][j0][i*wT+j]); 
      printf("\n");
    }
} //printLowerTileArray()


void initVec(int seed, double *x, int N) {
  int i;
  srand(seed);
  for (i=0; i < N; i++) {
    x[i] = (2.0 * rand() / RAND_MAX) - 1.0;
    assert (-1.0 <= x[i] && x[i] <= 1.0);
  }  
} //initVec()

void printVec(char *name, double *x, int N) {
  int i;
  printf("%s:", name);
  for (i=0; i < N; i++) 
    printf(doubleFmt, x[i]);
  printf("\n");
} //printVec()
