/******************************************************************************
 * create a dummy GPU context (an infinite loop) to generate context switches
 *****************************************************************************/
#include <cuda.h>
#include <stdint.h>

__global__ void 
loop()
{
  asm volatile(
  "L0:"
    "bra L0;"
  );
}

int 
main(int argc, char *argv[])
{
  loop<<<1, 1>>>();
  cudaDeviceSynchronize();
}

