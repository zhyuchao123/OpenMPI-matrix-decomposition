// parallel functions for supporting tiled Cholesky factorization 
// template written by Peter Strazdins, Feb 20 for COMP4300/8300 Assignment 1 
// v1.0 28/02/20

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <mpi.h>
#ifdef USE_MKL
#include <mkl_cblas.h>
#else
#include <cblas.h>
#endif
#include <math.h>
#include "auxCholesky.h"
#include <string.h>
//native Fortran subroutine
int dpotrf_(char *UpLo , int *N, double *A, int *ldA, int *info);

static int randDist;       // use random distribution; 
static int blockDist;      // use block distribution; 
static int seed;           // random number generator seed        
static int P, Q;           // if -b or no -r, use a PxQ process grid
static int optX;           // if set, initOwners() may assume choleskyX() 
                           // will be called 
static int verbosity = 0;  // output verbosity parameter          
static int tuneParam = 0;  // optional tuning parameter           
static int id, nprocs;     // MPI parameters  
static MPI_Comm comm;      // shorthand for MPI_COMM_WORLD

void initParams(int randDist_, int seed_, int blockDist_, int P_, int Q_, 
		int optX_, int verbosity_, int tuneParam_) {
  randDist = randDist_; seed = seed_; blockDist = blockDist_; P = P_, Q = Q_;
  optX = optX_; tuneParam = tuneParam_; verbosity = verbosity_;   
  MPI_Comm_rank(MPI_COMM_WORLD, &id);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  comm = MPI_COMM_WORLD;
} //initParams();



void initOwners(int **owner, int nT) {
  // for(int i=0;i <nT;i++){
  // Q = nprocs / P;
 if(blockDist==1 && randDist==0){
      // if(nT*nT <= nprocs){
      //   for(int i=0;i<nT;i++){
      //     for(int j=0;j<nT;j++){
      //       owner[i][j] = i*Q + j;
      //     }
      //   }
      // }else {
      //     int tile[P][Q];
      //     int rank = 0;
      //     for(int p=0;p<P;p++){
      //       for(int q=0;q<Q;q++){
      //         tile[p][q]=rank;
      //         rank +=1;
      //       }
      //     }
      //     int cseg= (nT%Q==0)? (nT/Q): (nT/Q)+1;
      //     int rseg = (nT%P==0)? nT/P: (nT/P)+1;
      //     int cstep = nT/cseg;
      //     int rstep = nT/rseg;

      //     for(int i=0;i<nT;i+=1){
      //     for(int j=0;j<nT;j+=1){
      //       // owner[i][j] = (i/cstep)*rseg + (j/cstep);
      //       owner[i][j] = tile[ (i/rseg)][j/cseg];

      //     }
      //   }
      // }
          int row = nT/P+1;
          int col = nT/Q +1;
          int tile[(nT/P)+1][(nT/Q)+1];
          int rank = 0;
          for(int p=0;p<row;p++){
            for(int q=0;q<col;q++){
              if (1==1)
              {
                /* code */
                tile[p][q]=rank % nprocs;
                rank +=1;
              }else{
                  tile[p][q]=0;

              }
            }
          }
    int i=0,j=0,color=0, tcor=0;
    for(int i=0;i<nT;i+=1){
      for(int j=0;j<nT;j+=1){
        // owner[i][j] = (i/cstep)*rseg + (j/cstep);
        owner[i][j] = tile[ (i/P)][j/Q];
      }
}

    
  // }
} //initOwners()
else if(blockDist==0 && randDist==1){
    // for random distribution
          for(int i=0;i<nT;i+=1){
          for(int j=0;j<nT;j+=1){
            // owner[i][j] = (i/cstep)*rseg + (j/cstep);
            owner[i][j] = rand()%nprocs;

          }
        }
      }else{
   for (int row = 0; row < nT; row++)
  {
    for (int col = 0; col < nT; col++)
    {
      /* code */
      owner[row][col] = (row%P) * Q + (col%Q);
    }
  }

}

  int total = nT*nT;
  for(int i=0; i< total;i++){
      int row = i/nT;
      int col = i % nT;
      if(col>row){
        owner[row][col]=-1;
      }
  }
}

// note: (info != 0) after dpotrf_(), it means either A was not +ve 
// definite, or an error has occurred previously in the factorization.
// "Upper" is used instead of "Lower" as dpotrf_() assumes column-major storage


void choleskyLazy(int **ownerIdTile, int nT, int wT, double ***A) {
  int i, j, k,o,p;

  for (k=0; k<nT; k++) {
    MPI_Status status;
    MPI_Request r1,r2,r3;
    if (id == ownerIdTile[k][k]) { //see above note on dpotrf_()
      int info = 0;
      dpotrf_("Upper", &wT, A[k][k], &wT, &info); //Cholesky factor A[k][k]


      if (info != 0){
	       printf("%d: WARNING: dpotrf() failed: tile (%d,%d), element %d=%.3f\n",
	       id, k, k, info, A[k][k][info]);
      } // end if info
    } // end id == ownerIdTile[k][k]
      
    if (A[k][k]==0)
      {
        // allocate A[k][k] memory space
        A[k][k] = (double *) malloc(sizeof(double)*wT*wT);
      }
      // once dpotrf update, broadcast to A[i][k]
      // where i>k before the cblas_dstrsm happens
      // but now we use boardcast to update the element
      if (id == ownerIdTile[k][k])/*for send*/
      {
    //     int countArr[nprocs];
    //     memset(countArr,0, nprocs);

    //     for ( o = 0; o < P; o++){
    //       for(p=0; p<Q;p++){
    //       // if (id!=ownerIdTile[o][p] && p==k && o >=k)
    //         countArr[ownerIdTile[o][p]]+=1;
    //           if (id!=ownerIdTile[o][p] && countArr[ownerIdTile[o][p]]<=1)

    //       {
    //         /* code */
    //           // MPI_Isend(A[k][k], wT*wT, MPI_DOUBLE,o, 1,comm,&r1);
    //         MPI_Send(A[k][k], wT*wT, MPI_DOUBLE, ownerIdTile[o][p], 1,comm);
    //       }
        
    //   }
    // }
        for(int proc=0;proc<nprocs;proc++){
          if (id!=proc){
             MPI_Send(A[k][k], wT*wT, MPI_DOUBLE, proc, 1,comm);

          }
        }
      // }else if((id-ownerIdTile[k][k])%Q==0 && id > id-ownerIdTile[k][k]==0 ){
  }else{
        //   for (int o = 0; o < P; o++){
        //     for(int p=0;p<Q; p++){
        //       if (id != ownerIdTile[o][p]){
        //           // int tag;
        //           MPI_Irecv(A[k][k], wT*wT, MPI_DOUBLE, ownerIdTile[k][k], 1 ,comm, &r1);
        //       }  
        //     }
        // }
            // MPI_Irecv(A[k][k], wT*wT, MPI_DOUBLE, ownerIdTile[k][k], 1 ,comm, &r1);
          MPI_Recv(A[k][k], wT*wT, MPI_DOUBLE, ownerIdTile[k][k], 1 ,comm,&status);
      }
      // MPI_Bcast(A[k][k],wT*wT, MPI_DOUBLE, ownerIdTile[k][k],comm);
    // MPI_Wait(&r1,&status);
    MPI_Barrier(comm);

    for (i=k+1; i<nT; i++) {

      if (id == ownerIdTile[i][k]) { //A[i][k] = A[i][k] * A[k][k]^-T
	      cblas_dtrsm(CblasRowMajor, CblasRight, CblasLower, CblasTrans, 
		    CblasNonUnit, wT, wT, 1.0, A[k][k], wT, A[i][k], wT);
        // it means 
      } // end id == ownerIdTile[i][k]
    } //end for(i=k+1; i<nT; i++)
    // we should boardcast dtrsm result to specific communicator 
    // for example 
    for(i=k+1;i<nT; i++){
       if (A[i][k]==0)
        {
          // allocate A[k][k] memory space 
          A[i][k] = (double *) malloc(sizeof(double)*wT*wT);
        }
        // for receive and send
        // MPI_Bcast(A[i][k],wT*wT, MPI_DOUBLE,ownerIdTile[i][k], comm);
        if (id == ownerIdTile[i][k])
        {
        // int countArr2[nprocs];
        // memset(countArr2,0, nprocs);

        //   for ( o = 0; o < P; o++){
        //     for(p = 0; p< Q; p++){
        //     int row = id /Q;
        //     int col = id % Q;
        //     countArr2[ownerIdTile[o][p]]+=1;
        //     if (id!=ownerIdTile[o][p] && countArr2[ownerIdTile[o][p]]<=1 /*&&((row == o && col = o))*/)
        //     {
        //         // MPI_Isend(A[i][k], wT*wT, MPI_DOUBLE,o, 1,comm,&r2);
        //         MPI_Send(A[i][k], wT*wT, MPI_DOUBLE,ownerIdTile[o][p], 1,comm);

        //     }
        //   }
        //   }
          for(int proc=0;proc<nprocs;proc++){
            if(id!=proc){
              MPI_Send(A[i][k], wT*wT, MPI_DOUBLE,proc, 1,comm);

            }
          }
        }else{
            // MPI_Irecv(A[i][k], wT*wT, MPI_DOUBLE,ownerIdTile[i][k], 1,comm,&r2);
          MPI_Recv(A[i][k], wT*wT, MPI_DOUBLE,ownerIdTile[i][k], 1,comm,&status);

        }
    }
    // MPI_Wait(&r2,&status);
    MPI_Barrier(comm);

    for (i=k+1; i<nT; i++) {
      for (j=k+1; j<=i; j++) {
      	if (id == ownerIdTile[i][j]) { // A[i][j] -=  A[i][k] * A[j][k]^T
      	  if (i==j) // only update lower tri. proportion of A[i][i]
      	    cblas_dsyrk(CblasRowMajor, CblasLower, CblasNoTrans, wT, wT, 
      			-1.0, A[i][k], wT, 1.0, A[i][i], wT);
      	  else
      	    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, wT, wT, wT, 
      			-1.0, A[i][k], wT, A[j][k], wT, 1.0, A[i][j], wT);
      	}
      } //for (j...) 
    } //for (i...) 
    // boardcast the result
    

  } //for (k...)

} //choleskyLazy()


void send_r1( int **ownerIdTile, int k, int nT, int wT,double ***A, int nonblk){
    // needTosend = (int) malloc(sizeof(int)* (nT-k)); // -1 is a flag to stop loop
    
    int *fullId = (int *) malloc(sizeof(int)*nprocs);
    memset(fullId,0,sizeof(int)*nprocs);
    int count = 0;
    int tmpid = 0;
    int senderId = ownerIdTile[k][k];
    if (A[k][k]==0)
      {
        // allocate A[k][k] memory space
        A[k][k] = (double *) malloc(sizeof(double)*wT*wT);
      }
    for(int j=k+1;j<=nT-1;j++){
      tmpid = ownerIdTile[j][k];
      if(fullId[tmpid]==0){
        fullId[tmpid] = 1;
        if ( tmpid!=ownerIdTile[k][k])
        {
            if(nonblk=0 ){
            
            MPI_Send(A[k][k], wT*wT, MPI_DOUBLE,tmpid, senderId*1000+k*nT+k,comm);

          }else{
            MPI_Request req;
            MPI_Isend(A[k][k], wT*wT, MPI_DOUBLE, tmpid , senderId*1000+k*nT+k, comm,&req);
            printf("send_r1, send A[j][k], j:%d k:%d\n",j,k );
          }
        }
        //needTosend[count]=id;
        count +=1;
      }
    }
    free(fullId);
    //needTosend[count+1]=-1; // ready to delete

}
void recv_r2( int **ownerIdTile, int k,int n, int nT, int wT,double ***A, int nonblk){
    int myid= ownerIdTile[n][k];
 
    if (n>=k+1 && n<=nT-1){
      // first receive item from A[k][k]


        if ( myid!=ownerIdTile[k][k])
        {
            if(nonblk=0 ){
            // MPI_Status status;
            MPI_Recv(A[k][k],wT*wT, MPI_DOUBLE,ownerIdTile[k][k],ownerIdTile[k][k]*1000+k*nT+k ,comm,MPI_STATUS_IGNORE);
          }else{
            MPI_Request req;
            MPI_Irecv(A[k][k], wT*wT, MPI_DOUBLE, ownerIdTile[k][k],ownerIdTile[k][k]*1000+k*nT+k, comm,&req);
            MPI_Wait(&req,MPI_STATUS_IGNORE);
          }
        }
      }
} //end recv_r2

void send_r2(int **ownerIdTile, int k,int n, int nT, int wT,double ***A, int nonblk){
    if (n>=k+1 && n<=nT-1){
    int myid= ownerIdTile[n][k];
    int *fullId = (int *) malloc(sizeof(int)*nprocs);
    memset(fullId,0,sizeof(int)*nprocs);
    int count = 0;
    int senderId = ownerIdTile[n][k];
    int tmpid = 0; // current id
    int j=0;
    printf("send_r2 in k:%d horizontal start##########\n",k);

    for(j=k+1;j<=n;j++){
      // for horizontal 
      tmpid = ownerIdTile[n][j];
      if(fullId[tmpid]==0){
        fullId[tmpid] = 1;

        if ( myid!= tmpid)
        {
            if(nonblk=0 ){
            
            MPI_Send(A[n][k], wT*wT, MPI_DOUBLE,tmpid, senderId*1000+n*nT+k,comm);

          }else{
            MPI_Request req;
            MPI_Isend(A[n][k], wT*wT, MPI_DOUBLE, tmpid , senderId*1000+n*nT+k, comm,&req);
          }
        }
        //needTosend[count]=id;
        //count +=1;
      }
    }


    printf("send_r2(senderId->tmpid) in k:%d horizontal pass###########\n",k);
    

    printf("send_r2 in k:%d vertical start............n",k);

    for( j=nT-1;j>n;j--){
      // for vertical down to up 
      tmpid = ownerIdTile[j][n];
      if(fullId[tmpid]==0){
        fullId[tmpid] = 1;

        if ( myid!= tmpid)
        {
            if(nonblk=0 ){
            MPI_Request req;
            MPI_Send(A[n][k], wT*wT, MPI_DOUBLE,tmpid, senderId*1000+n*nT+k,comm);
            //MPI_Wait(&req,MPI_STATUS_IGNORE);

            // if(n-1!=k){
            //   break;
            // }
          }else{
            MPI_Request req;
            MPI_Isend(A[n][k], wT*wT, MPI_DOUBLE, tmpid , senderId*1000+n*nT+k, comm,&req);
            //MPI_Wait(&req,MPI_STATUS_IGNORE);

            // if(n-1!=k){
            //   break;
            // }
        }
        //needTosend[count]=id;
        // count +=1;
      }
    }
  }
  free(fullId);
    }
}//end send_r2


void recv_r3(int **ownerIdTile, int k,int i,int j, int nT,int wT, double ***A, int nonblk){

  int myid = ownerIdTile[i][j];
  if(A[i][k]==0){
      A[i][k]== (double *) malloc(sizeof(double)*wT*wT);
    }
   if(A[j][k]==0){
        A[j][k]== (double *) malloc(sizeof(double)*wT*wT);
    }
   
  if (i==j)
  {
    if(nonblk==0 && ownerIdTile[i][k]!=myid){
      // block receive

      MPI_Recv(A[i][k],wT*wT, MPI_DOUBLE,ownerIdTile[i][k],ownerIdTile[i][k]*1000+i*nT+k ,comm,MPI_STATUS_IGNORE);

    }else if(nonblk== 1 && ownerIdTile[i][k]!=myid){
      //nonblock receive
      MPI_Request req;

      MPI_Irecv(A[i][k], wT*wT, MPI_DOUBLE, ownerIdTile[i][k],ownerIdTile[i][k]*1000+i*nT+k, comm,&req);
      MPI_Wait(&req,MPI_STATUS_IGNORE);


    }
  }else{

      if(nonblk==0 && ownerIdTile[j][k]!=myid){
            // block receive
            MPI_Recv(A[j][k],wT*wT, MPI_DOUBLE,ownerIdTile[j][k],ownerIdTile[j][k]*1000+j*nT+k ,comm,MPI_STATUS_IGNORE);

            MPI_Recv(A[i][k],wT*wT, MPI_DOUBLE,ownerIdTile[i][k],ownerIdTile[i][k]*1000+i*nT+k ,comm,MPI_STATUS_IGNORE);


          }else if(nonblk==1 && ownerIdTile[j][k]!=myid){
          //nonblock receive
          MPI_Request req1,req2;

          MPI_Irecv(A[j][k], wT*wT, MPI_DOUBLE, ownerIdTile[j][k],ownerIdTile[j][k]*1000+j*nT+k, comm,&req1);
          MPI_Wait(&req1,MPI_STATUS_IGNORE);

          MPI_Irecv(A[i][k], wT*wT, MPI_DOUBLE, ownerIdTile[i][k],ownerIdTile[i][k]*1000+i*nT+k, comm,&req2);
          MPI_Wait(&req2,MPI_STATUS_IGNORE);
      }

    }

}
void choleskyEager(int **ownerIdTile, int nT, int wT, double ***A) {
  int i, j, k,o,p;
  int *fullId;
  int nonblk = 1 ; // default blk send and recv
  for (k=0; k<nT; k++) {
    MPI_Status status;
    MPI_Request r1,r2,r3;


    if (id == ownerIdTile[k][k]) { //see above note on dpotrf_()
      int info = 0;
      dpotrf_("Upper", &wT, A[k][k], &wT, &info); //Cholesky factor A[k][k]
      printLowerTileArray(id, nT, wT, A);
      printf("r1, step %d \n",k);
      if (info != 0){
         printf("%d: WARNING: dpotrf() failed: tile (%d,%d), element %d=%.3f\n",
         id, k, k, info, A[k][k][info]);
      } // end if info

      // send_r1( ownerIdTile,k,nT,wT,A,nonblk);


    fullId = (int *) malloc(sizeof(int)*nprocs);
    memset(fullId,0,sizeof(int)*nprocs);
    // int count = 0;
    int tmpid = 0;
    int senderId = id;
    if (A[k][k]==0)
      {
        // allocate A[k][k] memory space
        A[k][k] = (double *) malloc(sizeof(double)*wT*wT);
      }
    for(int j=k+1;j<=nT-1;j++){
      tmpid = ownerIdTile[j][k];
      if(fullId[tmpid]==0){
        fullId[tmpid] = 1;
        if ( tmpid!=ownerIdTile[k][k])
        {
            MPI_Request req;
            MPI_Isend(A[k][k], wT*wT, MPI_DOUBLE, tmpid , senderId*1000+k*nT+k, comm,&req);
            printf("send_r1, send A[%d][%d]\n",j,k );
            MPI_Wait(&req,MPI_STATUS_IGNORE);
        }
        //needTosend[count]=id;
        // count +=1;
      }
    }
    free(fullId);
    } // end id == ownerIdTile[k][k]
      


  
    MPI_Barrier(comm);
    //MPI_Wait(&r1,&status);
    int n=0;
    for (n=k+1; n<nT; n++){
      if (id == ownerIdTile[n][k]){
        if (A[k][k]==0)
        {
          // allocate A[k][k] memory space 
          A[k][k] = (double *) malloc(sizeof(double)*wT*wT);
        }
        if(id!=ownerIdTile[k][k]){
            MPI_Request req;
            MPI_Irecv(A[k][k], wT*wT, MPI_DOUBLE, ownerIdTile[k][k],ownerIdTile[k][k]*1000+k*nT+k, comm,&req);
            MPI_Wait(&req,MPI_STATUS_IGNORE);
        }

      }
    }

    printf("id:%d r2 recv pass ############### for iteration k=%d\n",id,k);









    // MPI_Barrier(comm);
    printf("id:%d r2 send start ..........for iteration k=%d\n",id,k);

    for (i=k+1; i<nT; i++) {

      if (id == ownerIdTile[i][k]) { //A[i][k] = A[i][k] * A[k][k]^-T
        printf("before the dtrsm\n");
        cblas_dtrsm(CblasRowMajor, CblasRight, CblasLower, CblasTrans, 
        CblasNonUnit, wT, wT, 1.0, A[k][k], wT, A[i][k], wT);
        printf("after the dtrsm\n begin to call send_r2()");

        int myid= id;
        fullId = (int *) malloc(sizeof(int)*nprocs);
        memset(fullId,0,sizeof(int)*nprocs);
        int count = 0;
        int senderId = id;
        int tmpid = 0; // current id
        int j=0;
        printf("send_r2 in k:%d horizontal start##########\n",k);

        for(j=k+1;j<=i;j++){
          // for horizontal 
          tmpid = ownerIdTile[i][j];
          if(fullId[tmpid]==0){
            fullId[tmpid] = 1;

            if ( myid!= tmpid)
            {
                MPI_Request req;
                MPI_Isend(A[i][k], wT*wT, MPI_DOUBLE, tmpid , senderId*1000+i*nT+k, comm,&req);
            }
            //needTosend[count]=id;
            //count +=1;
          }
        }


        printf("send_r2(senderId->tmpid) in k:%d horizontal pass###########\n",k);
    

        printf("send_r2 in k:%d vertical start............n",k);

        for( j=nT-1;j>i;j--){
          // for vertical down to up 
          tmpid = ownerIdTile[j][i];
          if(fullId[tmpid]==0){
            fullId[tmpid] = 1;
            if ( myid!= tmpid)
            {
              
                MPI_Request req;
                MPI_Isend(A[i][k], wT*wT, MPI_DOUBLE, tmpid , senderId*1000+i*nT+k, comm,&req);
            }

          }
        }
        free(fullId);

        }

      } // end id == ownerIdTile[i][k]
     //end for(i=k+1; i<nT; i++)



  for (i=k+1; i<nT; i++) {
      for (j=k+1; j<=i; j++) {
        if (id == ownerIdTile[i][j]) {
          // recv_r3(ownerIdTile, k, i, j, nT,wT, A, nonblk); 
  int myid = id;
  if(A[i][k]==0){
      A[i][k]== (double *) malloc(sizeof(double)*wT*wT);
    }
   if(A[j][k]==0){
        A[j][k]== (double *) malloc(sizeof(double)*wT*wT);
    }
   
  if (i==j)
  {
    if(ownerIdTile[i][k]!=myid){
      //nonblock receive
      MPI_Request req;

      MPI_Irecv(A[i][k], wT*wT, MPI_DOUBLE, ownerIdTile[i][k],ownerIdTile[i][k]*1000+i*nT+k, comm,&req);
      MPI_Wait(&req,MPI_STATUS_IGNORE);


    }
  }else{

      if(ownerIdTile[j][k]!=myid){
          //nonblock receive
          MPI_Request req1,req2;

          MPI_Irecv(A[j][k], wT*wT, MPI_DOUBLE, ownerIdTile[j][k],ownerIdTile[j][k]*1000+j*nT+k, comm,&req1);
          MPI_Wait(&req1,MPI_STATUS_IGNORE);

          MPI_Irecv(A[i][k], wT*wT, MPI_DOUBLE, ownerIdTile[i][k],ownerIdTile[i][k]*1000+i*nT+k, comm,&req2);
          MPI_Wait(&req2,MPI_STATUS_IGNORE);
      }

    }//end recv3



        }
      }
    }




    for (i=k+1; i<nT; i++) {
      for (j=k+1; j<=i; j++) {
        if (id == ownerIdTile[i][j]) { // A[i][j] -=  A[i][k] * A[j][k]^T
          if (i==j) // only update lower tri. proportion of A[i][i]
            cblas_dsyrk(CblasRowMajor, CblasLower, CblasNoTrans, wT, wT, 
            -1.0, A[i][k], wT, 1.0, A[i][i], wT);
          else
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, wT, wT, wT, 
            -1.0, A[i][k], wT, A[j][k], wT, 1.0, A[i][j], wT);
        }
      } //for (j...) 
    } //for (i...) 
    // boardcast the result
    
  MPI_Barrier(comm);

  } //for (k...)
} //choleskyEager()



void choleskyX(int **ownerIdTile, int nT, int wT, double ***A) {
  int i, j, k;
  for (k=0; k<nT; k++) {

    if (id == ownerIdTile[k][k]) { //see above note on dpotrf_()
      int info = 0;
      dpotrf_("Upper", &wT, A[k][k], &wT, &info); //Cholesky factor A[k][k]
      // once dpotrf update, broadcast to A[i][k]
      // where i>k before the cblas_dstrsm happens
      // but now we use boardcast to update the element
      // if (A[k][k]==0)
      // {
      //   // allocate A[k][k] memory space
      //   A[k][k] = (double *) malloc(sizeof(double)*wT*wT);
      // }
      // // MPI_Comm subcomm = MPI_COMM_WORLD;
      // MPI_Bcast(A[k][k],wT*wT, MPI_DOUBLE, ownerIdTile[k][k],comm);

      if (info != 0){
         printf("%d: WARNING: dpotrf() failed: tile (%d,%d), element %d=%.3f\n",
         id, k, k, info, A[k][k][info]);
      } // end if info
    } // end id == ownerIdTile[k][k]
      
    if (A[k][k]==0)
      {
        // allocate A[k][k] memory space
        A[k][k] = (double *) malloc(sizeof(double)*wT*wT);
      }
      // MPI_Comm subcomm = MPI_COMM_WORLD;
      MPI_Bcast(A[k][k],wT*wT, MPI_DOUBLE, ownerIdTile[k][k],comm);

    for (i=k+1; i<nT; i++) {

      if (id == ownerIdTile[i][k]) { //A[i][k] = A[i][k] * A[k][k]^-T
        cblas_dtrsm(CblasRowMajor, CblasRight, CblasLower, CblasTrans, 
        CblasNonUnit, wT, wT, 1.0, A[k][k], wT, A[i][k], wT);
        // it means 
      } // end id == ownerIdTile[i][k]
    } //end for(i=k+1; i<nT; i++)
    // we should boardcast dtrsm result to specific communicator 
    // for example 
    for(i=k+1;i<nT; i++){
       if (A[i][k]==0)
        {
          // allocate A[k][k] memory space 
          A[i][k] = (double *) malloc(sizeof(double)*wT*wT);
        }
        // for receive and send
        MPI_Bcast(A[i][k],wT*wT, MPI_DOUBLE,ownerIdTile[i][k], comm);
    }

    for (i=k+1; i<nT; i++) {
      for (j=k+1; j<=i; j++) {
        if (id == ownerIdTile[i][j]) { // A[i][j] -=  A[i][k] * A[j][k]^T
          if (i==j) // only update lower tri. proportion of A[i][i]
            cblas_dsyrk(CblasRowMajor, CblasLower, CblasNoTrans, wT, wT, 
            -1.0, A[i][k], wT, 1.0, A[i][i], wT);
          else
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, wT, wT, wT, 
            -1.0, A[i][k], wT, A[j][k], wT, 1.0, A[i][j], wT);
        }
      } //for (j...) 
    } //for (i...) 
    // boardcast the result
    
    for (i=k+1; i<nT; i++) {
      for (j=k+1; j<=i; j++) {
          if (A[i][j]==0){
              A[i][j] = (double *) malloc(sizeof(double)*wT*wT);
          }
          if (i==j) // only update lower tri. proportion of A[i][i]
            // comm should be modified later
            MPI_Bcast(A[i][j], wT*wT,MPI_DOUBLE, ownerIdTile[i][j],comm);
          else
            MPI_Bcast(A[i][j], wT*wT,MPI_DOUBLE, ownerIdTile[i][j],comm);   
        } //for (j...) 
      }
  } //for (k...)
} //choleskyX()
