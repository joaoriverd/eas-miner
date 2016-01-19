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
#define SIZE_INPUT INPUT_SIZE+NONCE_SIZE+1
#define SIZE_OUTPUT 33

extern uint32_t* d_in;
extern uint32_t* d_a;
extern uint32_t* d_b;
extern char* d_input;
extern char* d_output;
extern uint32_t* d_p;

__global__ void RoundFunction(uint32_t* a, uint32_t* b);

__global__ void InputFunction(uint32_t* in, uint32_t* a, uint32_t* b);
__global__ void OutputFunction(uint32_t* out, uint32_t* a);

__global__ void inLoop(uint32_t* in, char* input, uint32_t* p);
__global__ void outLoop(uint32_t* out, char* output, uint32_t* i);

void Hash(char* input, char* output);


#endif
