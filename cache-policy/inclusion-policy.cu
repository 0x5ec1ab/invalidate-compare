/******************************************************************************
 * check inclusion policy of L2
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
check_inclusion(uint64_t addr, uint64_t *vals)
{
  uint64_t val_old;
  uint64_t val_new;
  
  uint32_t smid;
  asm volatile("mov.u32 %0, %%smid;\n\t" : "=r" (smid));
  
  if (smid != SMID_X && smid != SMID_Y)
    return;

  asm volatile(
    ".reg .u64 addr_reg;"
    "mov.u64 addr_reg, %1;"
    "ld.u64.ca %0, [addr_reg];"
    : "=l" (val_old) 
    : "l" (addr)
  );
  wait_delay(SHORT_DELAY);
  
  if (smid == SMID_X) {
    // perform "X writes B"
    asm volatile("st.u64.wb [addr_reg], 0xdeaddead;");
    
    // make sure "Y writes B" finished
    wait_delay(SHORT_DELAY);
    
    // perform "X discards B"
    asm volatile("discard.global.L2 [addr_reg], 128;");
    
    // make sure "Y reads B" finished
    wait_delay(LONG_DELAY);
    
    // perform "X reads B"
    asm volatile("ld.u64.ca %0, [addr_reg];" : "=l" (val_new));
    
    vals[0] = val_old;
    vals[1] = val_new;
  } else {
    // perform "Y writes B"
    asm volatile("st.u64.wb [addr_reg], 0xbeefbeef;");
    
    // make sure "X discards B" finished
    wait_delay(MID_DELAY);
    
    // perform "Y reads B"
    asm volatile("ld.u64.ca %0, [addr_reg];" : "=l" (val_new));
    
    vals[2] = val_old;
    vals[3] = val_new;
  }
}

int 
main(int argc, char *argv[])
{
  cudaDeviceReset();
  cudaFuncSetAttribute(check_inclusion, 
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
  check_inclusion<<<BLK_NUM, 1, SHARED_MEM>>>(addr, vals);
  cudaDeviceSynchronize();

  cudaMemcpy(host, vals, CHUNK_SIZE, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  printf("SM X: old value %lx \t new value %lx\n", host[0], host[1]);
  printf("SM Y: old value %lx \t new value %lx\n", host[2], host[3]);
  
  free(host);
  cudaFree(vals);
  cudaFree(data);
}

