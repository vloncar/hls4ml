#!/bin/bash

CC=g++
PYTHON_BIN_PATH=python

OP_SRC=layer_op.cpp

TF_CFLAGS=$(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS=$(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')

INCFLAGS=-Iap_types/

OMPFLAGS=-fopenmp

CFLAGS="${TF_CFLAGS} -fPIC -O2 -std=c++14 -Wno-deprecated-declarations ${OMPFLAGS} ${INCFLAGS}"
LDFLAGS="-shared ${TF_LFLAGS}"

TARGET_LIB=layer_op.so

${CC} ${CFLAGS} -o ${TARGET_LIB} ${OP_SRC} ${LDFLAGS}