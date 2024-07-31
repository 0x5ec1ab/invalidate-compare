/******************************************************************************
 * check write-allocate policy of L1
 *****************************************************************************/
#include <cuda.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

// a chunk consists of a 2MB page
#define CHUNK_SIZE   (2 * 1024 * 1024)

// force each block to reside at one SM
#define BLK_NUM     100
#define SHARED_MEM  (90 * 1024)

// two SMs chosen for this experiment
#define SMID_X      0
#define SMID_Y      12

#define SHORT_DELAY 1000000L
#define MID_DELAY   100000000L
#define LONG_DELAY  10000000000L

__device__ void 
wait_delay(uint64_t delay)
{
  uint64_t start;
  uint64_t diff;
  
  start = clock64();
  do {
    diff = clock64() - start;
  } while (diff < delay);
}

__global__ void 
check_l1_allocate(uint64_t addr, uint64_t *vals)
{
  uint64_t temp;
  
  uint32_t smid;
  asm volatile("mov.u32 %0, %%smid;\n\t" : "=r" (smid));
  
  if (smid != SMID_X && smid != SMID_Y)
    return;

  asm volatile(
    ".reg .u64 addr_reg;"
    "mov.u64 addr_reg, %0;"
    : 
    : "l" (addr)
  );
  
  if (smid == SMID_X) {
    // wait for "Y reads B"
    wait_delay(SHORT_DELAY);
    
    // perform "X writes B"
    asm volatile("st.u64 [addr_reg], 0xdeadbeef;");
    //asm volatile("st.global.L1::no_allocate.u64 [addr_reg], 0xdeadbeef;");
    
    // wait for "Y discards B"
    wait_delay(LONG_DELAY);
    
    // perform "X reads B"
    asm volatile("ld.u64 %0, [addr_reg];" : "=l" (temp));
    
    vals[0] = temp;
  } else {
    // perform "Y reads B"
    asm volatile("ld.u64 %0, [addr_reg];" : "=l" (temp));
    
    // wait for "X writes B"
    wait_delay(MID_DELAY);
    
    // perform "Y discards B"
    asm volatile("discard.global.L2 [addr_reg], 128;");
    
    vals[1] = temp;
  }
}

int 
main(int argc, char *argv[])
{
  cudaDeviceReset();
  cudaFuncSetAttribute(check_l1_allocate, 
      cudaFuncAttributeMaxDynamicSharedMemorySize, SHARED_MEM);
    
  uint64_t *data;
  uint64_t *vals;
  uint64_t *host;
  cudaMalloc(&data, CHUNK_SIZE);
  cudaMalloc(&vals, CHUNK_SIZE);
  cudaDeviceSynchronize();
  
  host = (uint64_t *)malloc(CHUNK_SIZE);
  memset(host, 0, CHUNK_SIZE);
  
  cudaMemcpy(data, host, CHUNK_SIZE, cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  
  uint64_t addr = (uint64_t)data;
  check_l1_allocate<<<BLK_NUM, 1, SHARED_MEM>>>(addr, vals);
  cudaDeviceSynchronize();

  cudaMemcpy(host, vals, CHUNK_SIZE, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  printf("SM X: value %lx\n", host[0]);
}

