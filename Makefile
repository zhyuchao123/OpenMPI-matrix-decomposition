# Makefile for COMP4300/8300 Assignment 1
# Peter Strazdins, RSCS ANU, Mar 20
.SUFFIXES:
.PRECIOUS: %.o

HDRS=auxCholesky.h checkCholesky.h parCholesky.h
OBJS=auxCholesky.o checkCholesky.o parCholesky.o
PROG=testCholesky

HOST=$(shell hostname | awk -F- '{print $$1}')
ifeq ($(HOST),gadi) # must use MKL BLAS
CC=icc
CCFLAGS=-O3 -DUSE_MKL -mkl -pg
LDFLAGS=-mkl 
else # assume OpenBLAS
CCFLAGS=-O3
LDFLAGS=-llapack -lblas 
endif

all: $(PROG) 

%: %.o $(OBJS)
	mpicc -o $* $*.o $(OBJS) $(LDFLAGS) -lm 
%.o: %.c $(HDRS)
	mpicc -Wall $(CCFLAGS) -c $*.c
clean:
	rm -f *.o $(PROG)
