#include <iostream>
#include <stdlib.h>
#include "cuda_utils.h"

int main(int argc, char const *argv[]){
    if( argc < 3){
        std::cerr << "Missing arguments." << std::endl
                << "Usage: ./rayTracer <h> <w>" << std::endl;
        exit(-1);
    }
    
    int imageWidth = atoi(argv[1]);
    int imageHeight = atoi(argv[2]);

    std:: cerr << "Rendering a " << imageWidth << "x" << imageHeight << " image in "
                << BLOCK_X << "x" << BLOCK_Y << " blocks.\n";


    return 0;
}