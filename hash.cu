#include <stdio.h>
#include <string.h>
#include "hash.cuh"

#define BW4 (sizeof(uint32_t)*BW)

__device__ void RoundFunction_CUDA(uint32_t* a, uint32_t* b);
__device__ void RF_L1(uint32_t *d_q, uint32_t *d_b);
__device__ void RF_L2(uint32_t *d_b);
__device__ void RF_L3(uint32_t *d_q,uint32_t *d_b);
__device__ void RF_L4(uint32_t *d_a,uint32_t *d_b);
__device__ void RF_L5(uint32_t *d_A,uint32_t *d_a);
__device__ void RF_L6(uint32_t *d_A,uint32_t *d_a);
__device__ void RF_L7(uint32_t *d_A,uint32_t *d_a);
__device__ void RF_L8(uint32_t *d_A,uint32_t *d_a);
__device__ void RF_L9(uint32_t *d_q,uint32_t *d_a);

__device__ inline unsigned int index2(unsigned int i, unsigned int j){
    return (unsigned int) (i*BW+j);
}

__device__ inline uint32_t ROR2(uint32_t x, int y){
    int y_mod = ((y & 0x1F) + 32) & 0x1F;
    return ROR32(x, y_mod);
}

//note: output must be 32+1 chars (+1 for termination of string)
__global__ void Hash(char* input, char* output, uint32_t* inputSize_in, uint32_t* debug)
{
    
    __shared__ uint32_t a[MS];
    __shared__ uint32_t b[BL*BW];
    __shared__ uint32_t in[BW];
    __shared__ uint32_t out[2];
    __shared__ unsigned int d_p;
    __shared__ unsigned int d_i;

    //init with zeros
    for(unsigned int i=0; i<MS; i++)
        a[i] = 0;
    for(unsigned int i=0; i<BL*BW; i++)
        b[i] = 0;
    
    uint32_t inputSize = inputSize_in[0];

    *debug = 0;
    uint32_t inputSize_norm = inputSize/BW4;
    unsigned int p = inputSize-inputSize%BW4;
    
     d_p = 0;
    for(unsigned int i=0; i<inputSize_norm; i++){
        inLoop(in,input,&d_p);
        InputFunction(in,a,b);
        RoundFunction(a,b);
        //RoundFunction_CUDA(a,b); // test LZAVALAM
        (*debug)++;
    }
    
    //*debug = a[0];//debug
  
    //padding
    //char* last_block = (char*) calloc(BW+1, sizeof(uint32_t));
    char last_block[(BW+1)*sizeof(uint32_t)];
    for(unsigned int i=0; i<(BW+1)*sizeof(uint32_t); i++)
        last_block[i] = 0;
    
    for(uint32_t i=0;i<inputSize%BW4;i++)
        last_block[i]=input[p+i];
    last_block[inputSize%BW4]=(char) 0x01;
    
    d_p = 0;
    inLoop(in,last_block,&d_p);
    InputFunction(in,a,b);
    RoundFunction(a,b);
    (*debug)++;
 
   //do some iterations without new input
    for(uint32_t i=0; i<BI; i++){
        RoundFunction(a,b);
        (*debug)++;
    }
    //*debug = a[0];//debug
    
    //collect 32 output characters
    d_i = 0;
    for(uint32_t i=0;i<32/(2*sizeof(uint32_t));i++){
        RoundFunction(a,b);
        (*debug)++;
        OutputFunction(out,a);
        outLoop(out, output, &d_i);
    }
    output[32]='\0';
 
}
    

__device__ void RoundFunction_CUDA(uint32_t* a, uint32_t* b)
{
    uint32_t q[BW];
    RF_L1(q,b);
    RF_L2(b);
    RF_L3(q,b);
    RF_L4(a,b);
   
    uint32_t A[MS];
    RF_L5(A,a);
    RF_L6(A,a);
    RF_L7(A,a);
    RF_L8(A,a);
    RF_L9(q,a);
}
__device__ void RoundFunction(uint32_t* a, uint32_t* b)
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
        a[i] = ROR2(A[(7*i)%MS], i*(i+1)/2);
    
    for(unsigned int i=0; i<MS; i++)
        A[i] = a[i]^a[(i+1)%MS]^a[(i+4)%MS];
   
    A[0] ^= 1;
   
    for(unsigned int i=0; i<MS; i++) 
        a[i] = A[i];

   
    for(unsigned int j=0; j<BW; j++)
        a[j+13] ^= q[j];
}

__device__ void InputFunction(uint32_t* in, uint32_t* a, uint32_t* b)
{  
    for(unsigned int j=0; j<BW; j++) 
        a[j+16] ^= in[j];
    
    for(unsigned int j=0; j<BW; j++) 
        b[index2(0,j)] ^= in[j];
}

__device__ void OutputFunction(uint32_t* out, uint32_t* a)
{
    for(unsigned int j=0; j<2; j++)
        out[j] = a[j+1];
}

__device__ void inLoop(uint32_t* in, char* input, uint32_t* p)
{   
    for(unsigned int q=0; q<BW; q++) {
            in[q] = 0;
            for(unsigned int w=0; w<sizeof(uint32_t); w++)
                in[q] |= (uint32_t)((unsigned char)(input[(*p)+q*sizeof(uint32_t)+w])) << (8*w);
    }
    (*p) += sizeof(uint32_t)*BW;
}

__device__ void outLoop(uint32_t* out, char* output, uint32_t* i)
{   
    for(unsigned int q=0; q<2; q++)
            for(unsigned int w=0; w<sizeof(uint32_t); w++)
                output[(*i)*sizeof(uint32_t)*2+q*sizeof(uint32_t)+w] = (char)((out[q] >> (8*w)) & 0xFF);
    (*i)++;
}



__device__ void RF_L1(uint32_t *d_q, uint32_t *d_b){
    int j =  threadIdx.x + blockIdx.x * blockDim.x;
    while(j<BW){
        d_q[j] = d_b[index2(BL-1,j)];
        j+= blockDim.x *gridDim.x;}
}
__device__ void RF_L2(uint32_t *d_b){
    for(unsigned int i=BL-1; i>0; i--)
        for(unsigned int j=0; j<BW; j++)
            d_b[index2(i,j)] = d_b[index2(i-1,j)];
}
__device__ void RF_L3(uint32_t *d_q,uint32_t *d_b){
    int j =  threadIdx.x + blockIdx.x * blockDim.x;
    while(j<BW){
        d_b[index2(0,j)] = d_q[j];
        j+= blockDim.x *gridDim.x;}
}
__device__ void RF_L4(uint32_t *d_a,uint32_t *d_b){
    int i =  threadIdx.x + blockIdx.x * blockDim.x;
    while(i<12){
        d_b[index2(i+1,i%BW)] ^= d_a[i+1];
        i+= blockDim.x *gridDim.x;}
}
__device__ void RF_L5(uint32_t *d_A,uint32_t *d_a){
    int i =  threadIdx.x + blockIdx.x * blockDim.x;
    while(i<MS){
        d_A[i] = d_a[i]^(d_a[(i+1)%MS]|(~d_a[(i+2)%MS]));
        i+= blockDim.x *gridDim.x;}
}
__device__ void RF_L6(uint32_t *d_A,uint32_t *d_a){
    int i =  threadIdx.x + blockIdx.x * blockDim.x;
    while(i<MS){
        d_a[i] = ROR2(d_A[(7*i)%MS], i*(i+1)/2);
        i+= blockDim.x *gridDim.x;}
}
__device__ void RF_L7(uint32_t *d_A,uint32_t *d_a){
    int i =  threadIdx.x + blockIdx.x * blockDim.x;
    while(i<MS){
        d_A[i] = d_a[i]^d_a[(i+1)%MS]^d_a[(i+4)%MS];
        i+= blockDim.x *gridDim.x;}
    d_A[0] ^= 1;
}

__device__ void RF_L8(uint32_t *d_A,uint32_t *d_a){
    int i =  threadIdx.x + blockIdx.x * blockDim.x;
    while(i<MS){
        d_a[i] = d_A[i];
        i+= blockDim.x *gridDim.x;}
}
__device__ void RF_L9(uint32_t *d_q,uint32_t *d_a){
    int j =  threadIdx.x + blockIdx.x * blockDim.x;
    while(j<BW){
        d_a[j+13] ^= d_q[j];
        j+= blockDim.x *gridDim.x;}
}


