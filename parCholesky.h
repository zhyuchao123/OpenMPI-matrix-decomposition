// parallel functions for supporting tiled Cholesky factorization 
// written by Peter Strazdins, Feb 20 for COMP4300/8300 Assignment 1 
// v1.0 28/02/20

// initializes parameters for the other functions below; 
// these may assume that this function has been called first.
// verbosity and tuneParam are optional debugging and tuning parameters.
void initParams(int randDist, int seed, int blockDist, int P, int Q,
                int optX, int verbosity, int tuneParam);

// sets ownerIdTile[nT][nT] to the MPI ranks of the tiles of a lower
// triangular matrix (upper triangular elements should be set to -1),
// according to a random (seeded by seed), if randDist was set,
// block, if blockDist was set, or otherwise a cyclic distribution. 
// For block and cyclic, a PxQ process grid is used, where p = nprocs / P;
// the ordering is row-major with respect to id.
// If optX was set, the distribution required by choleskyX() should be set; 
// this may be different (or the same) as the above distributions. 
void initOwners(int **ownerIdTile, int nT);


// in the following, ownerIdTile[nT][nT] contains the MPI ranks of the owners 
// of the nT x nT tiles of A, which are each of size wT x wT. 
// They may assume initOwners(ownerIdTile, nT) has been called 
// to set ownerIdTile[nT][nT].

// the following functions should obey the following performance constraints:
// 1. the same tile is never sent more than once between any pair of processes
//    (advise to get the code working first, then apply this optimization)
// 2. they should not leak memory, i.e. any malloced data is freed.
//    In particular, any buffers allocated for receiving messages should
//    be freed within 2 iterations of the outer (k) loop.


// uses `lazy' communication of tiles: a tile is sent/received just before
// it is needed as an input for a (TRSM, SYRK or GEMM) computation.
// `Just before' for the send() means at the same point in the k-i-j tile
// iteration space as for the recv().
void choleskyLazy(int **ownerIdTile, int nT, int wT, double ***A);

// uses `eager' communication of tiles: asynchronous send/recv calls are posted
// as soon as the tile has been updated by a (POTRF or TRSM) computation.
// `As soon' on the recv() means at the same point in the k-i-j tile
// iteration space as for the send().
void choleskyEager(int **ownerIdTile, int nT, int wT, double *** A);

// uses an optimized algorithm (possibly with a different distribution
// than random or PxQ cyclic/block) 
void choleskyX(int **ownerIdTile, int nT, int wT,  double ***A);

