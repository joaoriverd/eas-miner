#ifndef _HASH_H_
#define _HASH_H_
#include "common.h"
#include <stdint.h>
#include <stdlib.h>

void RoundFunction();

__global__ void InputFunction(uint32_t* in, uint32_t* d_a, uint32_t* d_b);
void OutputFunction(uint32_t* out);

void Hash(char* input, char* output);


#endif
