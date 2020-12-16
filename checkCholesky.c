// parallel functions for supporting tiled Cholesky factorization checking
// written by Peter Strazdins, Feb 20 for COMP4300/8300 Assignment 1 
// v1.1 19/03/20

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <mpi.h>
#include <string.h> //memset()
#ifdef USE_MKL
#include <mkl_cblas.h>
#else
#include <cblas.h>
#endif

#include "auxCholesky.h"

static int id;

void triMatVecMult(int nT, int wT, int **ownerIdTile, double ***A, 
		   double *x, double *y) {
  int i, j;
  MPI_Comm_rank(MPI_COMM_WORLD, &id);
  memset(y, 0, nT*wT*sizeof(double)); // y = 0.0;

  // y = A*x + y
  for (i=0; i<nT; i++) {
    for (j=0; j<nT; j++) { 
      // y[i*wT..i*wT+wT-1] += A[i][j] * x[j*wT..j*wT+wT-1] 
      if (i>j && id==ownerIdTile[i][j])
	cblas_dgemv(CblasRowMajor, CblasNoTrans, wT, wT,
		    1.0, A[i][j], wT, &x[j*wT], 1, 1.0, &y[i*wT], 1); 
      else if (i==j && id==ownerIdTile[i][i])
	cblas_dsymv(CblasRowMajor, CblasLower, wT, 
		    1.0, A[i][i], wT, &x[j*wT], 1, 1.0, &y[i*wT], 1);
      else if (i<j && id==ownerIdTile[j][i])
	cblas_dgemv(CblasRowMajor, CblasTrans, wT, wT,
		    1.0, A[j][i], wT, &x[j*wT], 1, 1.0, &y[i*wT], 1); 
    } //for (j...)
  } //for (i...)

  MPI_Allreduce(MPI_IN_PLACE, y, nT*wT, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
} //triMatVecMult()


void triMatVecSolve(int nT, int wT, int **ownerIdTile, double ***L, double *y) {
  double *yi;
  int i, j;
  MPI_Comm_rank(MPI_COMM_WORLD, &id);
  yi = (double *) malloc(sizeof(double) * wT);
  
  // y = L^-1 * y
  for (i=0; i<nT; i++) {
    memset(yi, 0, sizeof(double) * wT); // yi = 0.0;
    for (j=0; j < i; j++) { 
      // yi -= A[i][j] * y[j*wT..j*wT+wT-1] 
      if (id == ownerIdTile[i][j])	
	cblas_dgemv(CblasRowMajor, CblasNoTrans, wT, wT,
		    -1.0, L[i][j], wT, &y[j*wT], 1, 1.0, yi, 1); 
    } //for (j...)
    MPI_Reduce(id==ownerIdTile[i][i]? MPI_IN_PLACE: yi, yi, wT, MPI_DOUBLE, 
	       MPI_SUM, ownerIdTile[i][i], MPI_COMM_WORLD);
    
    if (id == ownerIdTile[i][i]) {
      cblas_daxpy(wT, 1.0, yi, 1, &y[i*wT], 1);  //y[i*wT..i*wT+wT-1] += yi
      //y[i*wT..i*wT+wT-1] = L[i][i]^-1 * y[i*wT..i*wT+wT-1]       
      cblas_dtrsv(CblasRowMajor, CblasLower, CblasNoTrans, CblasNonUnit, wT,
		  L[i][i], wT, &y[i*wT], 1);
    }
    MPI_Bcast(&y[i*wT], wT, MPI_DOUBLE, ownerIdTile[i][i], MPI_COMM_WORLD);
  } //for (i...)
   
  // y = L^-T * y
  for (i=nT-1; i>=0; i--) {
    memset(yi, 0, sizeof(double) * wT); // yi = 0.0;
    for (j=nT-1; j > i; j--) { 
      // yi -= A[j][i]^T * y[j*wT..j*wT+wT-1] 
      if (id == ownerIdTile[j][i])	
	cblas_dgemv(CblasRowMajor, CblasTrans, wT, wT,
		    -1.0, L[j][i], wT, &y[j*wT], 1, 1.0, yi, 1); 
    } //for (j...)
    MPI_Reduce(id==ownerIdTile[i][i]? MPI_IN_PLACE: yi, yi, wT, 
	       MPI_DOUBLE, MPI_SUM, ownerIdTile[i][i], MPI_COMM_WORLD);

    if (id == ownerIdTile[i][i]) {
      cblas_daxpy(wT, 1.0, yi, 1, &y[i*wT], 1);  //y[i*wT..i*wT+wT-1] += yi
      //y[i*wT..i*wT+wT-1] = L[i][i]^-T * y[i*wT..i*wT+wT-1]       
      cblas_dtrsv(CblasRowMajor, CblasLower, CblasTrans, CblasNonUnit, wT,
		  L[i][i], wT, &y[i*wT], 1);
    }
    MPI_Bcast(&y[i*wT], wT, MPI_DOUBLE, ownerIdTile[i][i], MPI_COMM_WORLD);
  } //for (i...)
   
  free(yi);
} //triMatVecSolve() 
