#ifndef _HASH_H_
#define _HASH_H_
#include "common.h"
#include <stdint.h>
#include <stdlib.h>

void RoundFunction();

void InputFunction(uint32_t* in);
void OutputFunction(uint32_t* out);

void Hash(char* input, char* output);


#endif
