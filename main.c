#include "hash.h"
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include "common.h"
#include "interface.h"

bool next_nonce(char *c){
    if((*c)=='\0'){
        return false;
    }
    
    //if end of range, wrap around and increment next 'digit'
    if((*c)=='9'){
        (*c)='a';
        if (next_nonce(c+sizeof(char)))
            return true;
        (*c)='\0';
        return false;
    }   

    //jump boundaries
    if((*c)=='z')
        (*c)='A';
    else if((*c)=='Z')
        (*c)='0';
    else
        (*c)++;

    return true;
}

bool check_hash(char* hash){
    //check if first character is a zero
    for(int i=0;i<LEADING_ZEROES;i++)
        if (hash[i]!='0')
            return false;
    return true;
}


void benchmark(void){
    // + 1 for termination '\0'
    char input[INPUT_SIZE+NONCE_SIZE+1];
    
    //position of repeated string in input
    char* base = &(input[NONCE_SIZE]);

    //holder for the nonce
    char nonce[NONCE_SIZE+1];

    while(true){
    
        //request new input from server (should be successful, o.w. just retry)
        while(!requestInput(base));

        //init nonce with 'a'*NONCE_SIZE
        for(int i=0;i<NONCE_SIZE;i++)
            nonce[i]='a';
        nonce[NONCE_SIZE]='\0';
        
        //test all possible nonces 
        do{
            //copy nonce into input
            for(int i=0;i<NONCE_SIZE;i++)
                input[i]=nonce[i];

            //32 chars + '\0' for binary output
            char output_hash[33];
            
            //calculate hash
            Hash(input, output_hash);

            //convert binary hash to printable hex
            char output_str[65];
            stringtohex_BE(output_hash,output_str);

            //check if hash matches desired output
            //if so, stop the search
            if (check_hash(output_str))
                break;

        }while(next_nonce(nonce));

        //if we reach here, either we found a matching nonce, or we exhaustively tested all nonces

        //validate with server
        validateHash(base, nonce);
    }

}

int main(int argc, char *argv[]){
    
   	if ((argc==2) && (strcmp(argv[1],"-benchmark")==0) ){
        benchmark();
    }

    else if (argc==4){
        //32 chars + 1 termination '\0'
        char output_hash[33];
        
        char* baseInput = argv[1];
        int baseInputSize=strlen(baseInput);
        int muliplier=atoi(argv[2]);
        char* nonce = argv[3];
        int nonce_size=strlen(nonce);

        //nonce first and append input string desired number of times
        char* input = (char*)malloc(sizeof(char)*(baseInputSize*muliplier+nonce_size+1));
        for(int i=0;i<nonce_size;i++)
            input[i]=nonce[i];
        char* repeat_ptr=&(input[nonce_size]);
        for(int j=0;j<muliplier;j++)
            for(int i=0;i<baseInputSize;i++)
                repeat_ptr[j*baseInputSize+i]=baseInput[i];
        input[baseInputSize*muliplier+nonce_size]='\0';

        //do hash
        Hash(input, output_hash);
        
        //convert binary hash to printable hex
        char output[65];
        stringtohex_BE(output_hash,output);

        printf("%s\n",output);

        free(input);

    }else{
        printf("usage: %s input(string) multiplier(int) nonce(string)\n", argv[0]);
        printf("------------OR-------------\n");
        printf("usage: %s -benchmark\n", argv[0]);
    }

	return 0;
}

