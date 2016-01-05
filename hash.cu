//#include "hash.cuh"
#include "hash.cuh"
#include <stdio.h>
#include <string.h>

#define MS  23
#define BW  5
#define BL  48
#define BI  17

#define SIZE_A MS*sizeof(uint32_t)
#define SIZE_B BL*BW*sizeof(uint32_t)
#define SIZE_IN BW*sizeof(uint32_t)

uint32_t a[MS];
uint32_t b[BL*BW];

   
inline unsigned int index2(unsigned int i, unsigned int j){
    return (unsigned int) (i*BW+j);
}

__device__ inline unsigned int index1(unsigned int i, unsigned int j){
    return (unsigned int) (i*BW+j);
}

//note: output must be 32+1 chars (+1 for termination of string)
void Hash(char* input, char* output)
{
    uint32_t in[BW];
    uint32_t out[2];
    
    /*****************************/
    
    dim3 threads,grid; //JR
    
    // utilities //JR
    //cudaEvent_t start;
    //cudaEvent_t stop;
    //float msecTotal;
    
    // Allocate memory for device 
    uint32_t* d_a;
    cudaMalloc((void**) &d_a, SIZE_A);
    uint32_t* d_b;
    cudaMalloc((void**) &d_b, SIZE_B);
    uint32_t* d_in;
    cudaMalloc((void**) &d_in, SIZE_IN);
    
    // create and start timer
    //cudaEventCreate(&start);
    //cudaEventRecord(start, NULL);
    
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
        
        // copy host memory to device //JR
        cudaMemcpy(d_in, in, SIZE_IN , cudaMemcpyHostToDevice);
        
        // call gpu
        InputFunction<<< grid, threads >>>(d_in, d_a, d_b);
        //InputFunction(in);
        
        // copy result from device to host
        cudaMemcpy(a, d_a, SIZE_A, cudaMemcpyDeviceToHost);
        cudaMemcpy(b, d_b, SIZE_B, cudaMemcpyDeviceToHost);
        
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
    
    // copy host memory to device //JR
    cudaMemcpy(d_in, in, SIZE_IN , cudaMemcpyHostToDevice);
    
    // call gpu
   InputFunction<<< grid, threads >>>(d_in, d_a, d_b);
   //InputFunction(in);
    
    // copy result from device to host
    cudaMemcpy(a, d_a, SIZE_A, cudaMemcpyDeviceToHost);
    cudaMemcpy(b, d_b, SIZE_B, cudaMemcpyDeviceToHost);
    
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
    
    // stop and destroy timer //JR
    //cudaEventCreate(&stop);
    //cudaEventRecord(stop, NULL);
    //cudaEventSynchronize(stop);
    //cudaEventElapsedTime(&msecTotal, start, stop);
    
    //free memory
    cudaFree(d_in);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaThreadExit();
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

/*
void InputFunction(uint32_t* in)
{
   
    for(unsigned int j=0; j<BW; j++) 
        a[j+16] ^= in[j];
    
    for(unsigned int j=0; j<BW; j++) 
        b[index2(0,j)] ^= in[j];
}
*/

__global__ void InputFunction(uint32_t* in, uint32_t* d_a, uint32_t* d_b)
{
    unsigned int j = threadIdx.x; 

    //for(unsigned int j=0; j<BW; j++) 
        d_a[j+16] ^= in[j];
    
    //for(unsigned int j=0; j<BW; j++) 
        d_b[index1(0,j)] ^= in[j];
}

void OutputFunction(uint32_t* out)
{
    for(unsigned int j=0; j<2; j++)
        out[j] = a[j+1];
}