#include "hash.h"
#include <stdio.h>
#include <string.h>

#define MS  23
#define BW  5
#define BL  48
#define BI  17

uint32_t a[MS];
uint32_t b[BL*BW];

   
inline unsigned int index2(unsigned int i, unsigned int j){
    return (unsigned int) (i*BW+j);
}

//note: output must be 32+1 chars (+1 for termination of string)
void Hash(char* input, char* output)
{
    uint32_t in[BW];
    uint32_t out[2];
   
    uint32_t inputSize=(uint32_t)strlen(input);

    //init with zeros
    for(unsigned int i=0; i<MS; i++)
        a[i] = 0;
    for(unsigned int i=0; i<BL*BW; i++)
        b[i] = 0;

    unsigned int p = 0;
    while(p+sizeof(uint32_t)*BW <=inputSize) {
        for(unsigned int q=0; q<BW; q++) {
            in[q] = 0;
            for(unsigned int w=0; w<sizeof(uint32_t); w++)
                in[q] |= (uint32_t)((unsigned char)(input[p+q*sizeof(uint32_t)+w])) << (8*w);
        }
        p += sizeof(uint32_t)*BW;
        InputFunction(in);
        RoundFunction();
    }
    
    //padding
    char* last_block = (char*) calloc(BW+1, sizeof(uint32_t));
    for(uint32_t i=0;i<inputSize-p;i++)
        last_block[i]=input[p+i];
    last_block[inputSize-p]=(char) 0x01;
    
    for(unsigned int q=0; q<BW; q++) {
        in[q] = 0;
        for(unsigned int w=0; w<sizeof(uint32_t); w++)
            in[q] |= (uint32_t)((unsigned char)(last_block[q*sizeof(uint32_t)+w])) << (8*w);
    }
    free(last_block);
    
    InputFunction(in);
    RoundFunction();
   
   //do some iterations without new input
    for(uint32_t i=0; i<BI; i++)
        RoundFunction();
    
    //collect 32 output characters
    for(uint32_t i=0;i<32/(2*sizeof(uint32_t));i++){
        RoundFunction();
        OutputFunction(out);

        for(unsigned int q=0; q<2; q++)
            for(unsigned int w=0; w<sizeof(uint32_t); w++)
                output[i*sizeof(uint32_t)*2+q*sizeof(uint32_t)+w] = (char)((out[q] >> (8*w)) & 0xFF);
    }
    output[32]='\0';
}


void RoundFunction()
{
    uint32_t q[BW];
    for(unsigned int j=0; j<BW; j++)
        q[j] = b[index2(BL-1,j)];

    for(unsigned int i=BL-1; i>0; i--)
        for(unsigned int j=0; j<BW; j++)
            b[index2(i,j)] = b[index2(i-1,j)];
    
    for(unsigned int j=0; j<BW; j++)
        b[index2(0,j)] = q[j];

    
    for(unsigned int i=0; i<12; i++)
        b[index2(i+1,i%BW)] ^= a[i+1];

   
    uint32_t A[MS];
    
    for(unsigned int i=0; i<MS; i++)
        A[i] = a[i]^(a[(i+1)%MS]|(~a[(i+2)%MS]));
   
    for(unsigned int i=0; i<MS; i++)
        a[i] = ROR(A[(7*i)%MS], i*(i+1)/2);
    
    for(unsigned int i=0; i<MS; i++)
        A[i] = a[i]^a[(i+1)%MS]^a[(i+4)%MS];
   
    A[0] ^= 1;
   
    for(unsigned int i=0; i<MS; i++) 
        a[i] = A[i];

   
    for(unsigned int j=0; j<BW; j++)
        a[j+13] ^= q[j];
}

void InputFunction(uint32_t* in)
{
   
    for(unsigned int j=0; j<BW; j++) 
        a[j+16] ^= in[j];
    
    for(unsigned int j=0; j<BW; j++) 
        b[index2(0,j)] ^= in[j];
}

void OutputFunction(uint32_t* out)
{
    for(unsigned int j=0; j<2; j++)
        out[j] = a[j+1];
}