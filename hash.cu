#include <stdio.h>
#include <string.h>

#define INPUT_FUNC_GPU

#ifdef INPUT_FUNC_GPU
 #include "hash.cuh"
#else
 #include "hash.h"
#endif

uint32_t a[MS];
uint32_t b[BL*BW];

#ifdef INPUT_FUNC_GPU
__device__ inline unsigned int index2(unsigned int i, unsigned int j){
    return (unsigned int) (i*BW+j);
}
#else
inline unsigned int index2(unsigned int i, unsigned int j){
    return (unsigned int) (i*BW+j);
}   
#endif

__device__ inline uint32_t ROR2(uint32_t x, int y){
    int y_mod = ((y & 0x1F) + 32) & 0x1F;
    return ROR32(x, y_mod);
}

//note: output must be 32+1 chars (+1 for termination of string)
void Hash(char* input, char* output)
{
    uint32_t in[BW];
    uint32_t out[2];
    
    /*****************************/
    
    dim3 threads,grid; //JR
    
    // setup execution parameters
    threads = dim3(BW, 1); // dummy fix this //JR
    grid = dim3(1,1); // dummy fix this //JR
    
    /*****************************/
   
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
#ifdef INPUT_FUNC_GPU        
        // copy host memory to device //JR
        cudaMemcpy(d_in, in, SIZE_IN , cudaMemcpyHostToDevice);
        
        // call gpu
        InputFunction<<< grid, threads >>>(d_in, d_a, d_b);
        RoundFunction<<< 1,1 >>>(d_a, d_b);
#else
        InputFunction(in);
        RoundFunction();
#endif  
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
#ifdef INPUT_FUNC_GPU
    // copy host memory to device //JR
    cudaMemcpy(d_in, in, SIZE_IN , cudaMemcpyHostToDevice);
    
    // call gpu
    InputFunction<<< grid, threads >>>(d_in, d_a, d_b);
    RoundFunction<<< 1,1 >>>(d_a, d_b);
#else
    InputFunction(in);
    RoundFunction();
#endif
 
   //do some iterations without new input
#ifdef INPUT_FUNC_GPU
    for(uint32_t i=0; i<BI; i++)
        RoundFunction<<< 1,1 >>>(d_a, d_b);
#else
    for(uint32_t i=0; i<BI; i++)
        RoundFunction();
#endif
    
    //collect 32 output characters
    for(uint32_t i=0;i<32/(2*sizeof(uint32_t));i++){
#ifdef INPUT_FUNC_GPU
        RoundFunction<<< 1,1 >>>(d_a, d_b);
        
        // copy result from device to host
        cudaMemcpy(a, d_a, SIZE_A, cudaMemcpyDeviceToHost);
#else
        RoundFunction();
#endif
        
        OutputFunction(out);

        for(unsigned int q=0; q<2; q++)
            for(unsigned int w=0; w<sizeof(uint32_t); w++)
                output[i*sizeof(uint32_t)*2+q*sizeof(uint32_t)+w] = (char)((out[q] >> (8*w)) & 0xFF);
    }
    output[32]='\0';

#ifdef TIMER
    // stop and destroy timer //JR
    cudaEventCreate(&stop);
    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&msecTotal, start, stop);
    printf("Processing time: %f (ms)\n", msecTotal);
#endif
}

#ifdef INPUT_FUNC_GPU
__global__ void RoundFunction(uint32_t* d_a, uint32_t* d_b)
{
    uint32_t q[BW];
    for(unsigned int j=0; j<BW; j++)
        q[j] = d_b[index2(BL-1,j)];

    for(unsigned int i=BL-1; i>0; i--)
        for(unsigned int j=0; j<BW; j++)
            d_b[index2(i,j)] = d_b[index2(i-1,j)];
    
    for(unsigned int j=0; j<BW; j++)
        d_b[index2(0,j)] = q[j];

    
    for(unsigned int i=0; i<12; i++)
        d_b[index2(i+1,i%BW)] ^= d_a[i+1];

   
    uint32_t A[MS];
    
    for(unsigned int i=0; i<MS; i++)
        A[i] = d_a[i]^(d_a[(i+1)%MS]|(~d_a[(i+2)%MS]));
   
    for(unsigned int i=0; i<MS; i++)
        d_a[i] = ROR2(A[(7*i)%MS], i*(i+1)/2);
    
    for(unsigned int i=0; i<MS; i++)
        A[i] = d_a[i]^d_a[(i+1)%MS]^d_a[(i+4)%MS];
   
    A[0] ^= 1;
   
    for(unsigned int i=0; i<MS; i++) 
        d_a[i] = A[i];

   
    for(unsigned int j=0; j<BW; j++)
        d_a[j+13] ^= q[j];
}

__global__ void InputFunction(uint32_t* in, uint32_t* d_a, uint32_t* d_b)
{
    unsigned int j = threadIdx.x; 
    
    //for(unsigned int j=0; j<BW; j++) 
        d_a[j+16] ^= in[j];
    
    //for(unsigned int j=0; j<BW; j++)
        d_b[j] ^= in[j];
}

#else
    
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
#endif

void OutputFunction(uint32_t* out)
{
    for(unsigned int j=0; j<2; j++)
        out[j] = a[j+1];
}