#ifndef _HASH_H_
#define _HASH_H_
#include "common.h"
#include <stdint.h>
#include <stdlib.h>

#define MS  23
#define BW  5
#define BL  48
#define BI  17

#define SIZE_A MS*sizeof(uint32_t)
#define SIZE_B BL*BW*sizeof(uint32_t)
#define SIZE_IN BW*sizeof(uint32_t)
#define SIZE_B BL*BW*sizeof(uint32_t)
#define SIZE_OUT 2*sizeof(uint32_t)

extern uint32_t* d_in;
extern uint32_t* d_a;
extern uint32_t* d_b;

__device__ void RoundFunction(uint32_t* a, uint32_t* b);

__device__ void InputFunction(uint32_t* in, uint32_t* a, uint32_t* b);
__device__ void OutputFunction(uint32_t* out, uint32_t* a);

__global__ void Hash(char* input, char* output, uint32_t* inputSize_in, uint32_t* debug);


#endif
