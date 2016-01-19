#include <stdio.h>
#include <string.h>
#include "hash.cuh"

__device__ inline unsigned int index2(unsigned int i, unsigned int j){
    return (unsigned int) (i*BW+j);
}

__device__ inline uint32_t ROR2(uint32_t x, int y){
    int y_mod = ((y & 0x1F) + 32) & 0x1F;
    return ROR32(x, y_mod);
}

//note: output must be 32+1 chars (+1 for termination of string)
void Hash(char* input, char* output)
{
    
    //uint32_t a[MS];
    //uint32_t b[BL*BW];
    //uint32_t in[BW];
    //uint32_t out[2];
    //unsigned int d_i;
    
    // copy host memory to device //JR
    cudaMemcpy(d_input, input, SIZE_INPUT , cudaMemcpyHostToDevice);

    //init with zeros
    cudaMemset(d_a,0,SIZE_A);
    cudaMemset(d_b,0,SIZE_B);
    cudaMemset(d_p,0,sizeof(uint32_t));
    
    uint32_t inputSize= (uint32_t)strlen(input);

    unsigned int p = 0;
    while(p+sizeof(uint32_t)*BW <=inputSize) {
        //inLoop<<<1,1>>>(d_in, d_input, d_p);
        p += sizeof(uint32_t)*BW;
        //InputFunction<<<1,1>>>(d_in,d_a,d_b);
        RoundFunction<<<1,1>>>(d_a,d_b);
    }
#if 0    
    //*debug = a[0];//debug
  
    //padding
    //char* last_block = (char*) calloc(BW+1, sizeof(uint32_t));
    char last_block[(BW+1)*sizeof(uint32_t)];
    for(unsigned int i=0; i<(BW+1)*sizeof(uint32_t); i++)
        last_block[i] = 0;
    
    for(uint32_t i=0;i<inputSize-p;i++)
        last_block[i]=input[p+i];
    last_block[inputSize-p]=(char) 0x01;
    
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
#endif
}
    
__global__ void RoundFunction(uint32_t* a, uint32_t* b)
{
#if 0
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
#endif 
}

__global__ void InputFunction(uint32_t* in, uint32_t* a, uint32_t* b)
{  
    for(unsigned int j=0; j<BW; j++) 
        a[j+16] ^= in[j];
    
    for(unsigned int j=0; j<BW; j++) 
        b[index2(0,j)] ^= in[j];
}

__global__ void OutputFunction(uint32_t* out, uint32_t* a)
{
    for(unsigned int j=0; j<2; j++)
        out[j] = a[j+1];
}

__global__ void inLoop(uint32_t* in, char* input, uint32_t* p)
{   
    for(unsigned int q=0; q<BW; q++) {
            in[q] = 0;
            for(unsigned int w=0; w<sizeof(uint32_t); w++)
                in[q] |= (uint32_t)((unsigned char)(input[(*p)+q*sizeof(uint32_t)+w])) << (8*w);
    }
    (*p) += sizeof(uint32_t)*BW;
}

__global__ void outLoop(uint32_t* out, char* output, uint32_t* i)
{   
    for(unsigned int q=0; q<2; q++)
            for(unsigned int w=0; w<sizeof(uint32_t); w++)
                output[(*i)*sizeof(uint32_t)*2+q*sizeof(uint32_t)+w] = (char)((out[q] >> (8*w)) & 0xFF);
    (*i)++;
}