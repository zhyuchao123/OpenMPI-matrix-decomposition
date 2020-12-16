// parallel functions for supporting tiled Cholesky factorization checking
// written by Peter Strazdins, Feb 20 for COMP4300/8300 Assignment 1 
// v1.0 27/02/20

//pre:  A stores the local tiles of a global symmetric matrix stored in the
//      lower triangular array of nT x nT tiles of size wT; 
//      ownerIdTile[i][j] determines which tiles are local
//      x and y are replicated vectors of size nT*wT
//post: y = A*x                              
void triMatVecMult(int nT, int wT, int **ownerIdTile, double ***A, 
		   double *x, double *y); 

//pre:  L stores the local tiles of a global lower triangular array of 
//      nT x nT tiles of size wT; 
//      ownerIdTile[i][j] determines which tiles are local
//      y is a replicated vector of size nT*wT
//post: y = (L^-1)^T * (L^-1) * y;
void triMatVecSolve(int nT, int wT, int **ownerIdTile, double ***L, double *y);
