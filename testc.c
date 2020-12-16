#include <stdio.h>
#include <string.h>
#include <stdlib.h>
int main(int argc, char const *argv[])
{
	/* code */
	int *A;
	int len = 5;
	A = (int*) malloc(sizeof(int)*5);	
	memset(A,-1,sizeof(int)*5);

	for (int i = 0; i < len ; i++)
	{
		/* code */
		printf("A is %i \n", A[i]);

	}
	free(A);
	A = (int*) malloc(sizeof(int)*4);	
}