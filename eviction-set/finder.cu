#include <cuda.h>
#include <unistd.h>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <vector>

#define CACHE_SIZE  (5 * 1024 * 1024)
#define BLK_SIZE    128

#define TARGET_NUM  256
#define START_POS   0

#define PAD_SIZE    (296 * 1024 * 1024)
#define CHUNK_SIZE  (64 * 1024 * 1024)

#define EVICTED     0x0000aaaa
#define NONEVICTED  0x0000bbbb

/******************************************************************************
 * 
 *****************************************************************************/
__global__ void 
check_eviction(uint64_t addr, uint64_t val, uint64_t *chain)
{
  uint64_t temp;
    
  asm volatile(
    ".reg .u64 addr_reg;"
    ".reg .u64 val_reg;"
    ".reg .u64 base_reg;"
    ".reg .u64 curr_reg;"
    ".reg .pred %p;"
    "mov.u64 addr_reg, %0;"
    "mov.u64 val_reg, %1;"
    "mov.u64 base_reg, %2;"
    "mov.u64 curr_reg, %2;"
    : 
    : "l" (addr), "l" (val), "l" (chain)
  );
  
  asm volatile(
    "st.u64 [addr_reg], val_reg;"
  "L0:"
    "st.u64 [curr_reg + 8], curr_reg;"
    "ld.u64 curr_reg, [curr_reg];"
    "setp.eq.u64 %p, curr_reg, base_reg;"
    "@!%p bra L0;"
  );
  
  // invalidate first
  asm volatile(
    "discard.global.L2 [addr_reg], 128;"
    "ld.u64.cg %0, [addr_reg];"
    : "=l" (temp)
  );
  
  // compare values
  if (temp == val)
    chain[0] = EVICTED;
  else
    chain[0] = NONEVICTED;
}

/******************************************************************************
 * 
 *****************************************************************************/
int 
main(int argc, char *argv[])
{
  uint64_t *pad;
  uint64_t *chunk;
  uint64_t *host;
  
  cudaMalloc(&pad, PAD_SIZE);
  cudaMalloc(&chunk, CHUNK_SIZE);
  cudaDeviceSynchronize();
  host = new uint64_t[CHUNK_SIZE / sizeof(uint64_t)];
  
  if (argc == 1) {
    std::cout << "chunk's virtual address: " << chunk << std::endl;
    std::cout << "before pressing Enter, go dump GPU memory..." << std::endl;
    std::getchar();
    
    std::cout << "adjust chunk's phys addr via PAD_SIZE, and run" << std::endl;
    std::cout << argv[0] << " <file to save eviction sets>" << std::endl;
    return 1;
  }
  
  std::ofstream res_file(argv[1]);
  if (!res_file) {
    std::cerr << argv[0] << " <file to save eviction sets>" << std::endl;
    return 1; 
  }
  
  uint64_t blk_num = (2 * CACHE_SIZE) / BLK_SIZE;
  assert(START_POS < blk_num - 1);
  
  for (uint64_t target = 0; target < TARGET_NUM; ++target) {
    std::set<uint64_t> res_set;
    uint64_t target_idx = (target * BLK_SIZE) / sizeof(uint64_t);
    uint64_t addr = (uint64_t)(chunk + target_idx);
    uint64_t val = 0x00000000deadbeef;
    
    int64_t mid = 0;
    int64_t lower = START_POS;
    int64_t upper = START_POS + blk_num - 1;
    
    std::cout << "==================== target: " << target << std::endl;
    while (upper >= lower) {
      // the mid guy (+1 guarantees ceiling)
      mid = (upper + lower + 1) / 2;
      
      std::vector<uint64_t> blk_vec(res_set.begin(), res_set.end());
      for (int64_t i = START_POS; i < mid; ++i) {
        if (res_set.count(i) != 0 || i == target)
          continue;
        blk_vec.push_back(i);
      }
      std::sort(blk_vec.begin(), blk_vec.end());
      
      for (uint64_t i = 0; i < blk_vec.size(); ++i) {
        uint64_t j = (i + 1) % blk_vec.size();
        uint64_t x = (blk_vec[i] * BLK_SIZE) / sizeof(uint64_t);
        uint64_t y = (blk_vec[j] * BLK_SIZE) / sizeof(uint64_t);
        host[x] = (uint64_t)(chunk + y);
      }
      cudaMemcpy(chunk, host, CHUNK_SIZE, cudaMemcpyHostToDevice);
      cudaDeviceSynchronize();
      
      uint64_t first_idx = (blk_vec[0] * BLK_SIZE) / sizeof(uint64_t);
      uint64_t *chain = chunk + first_idx;
      
      check_eviction<<<1, 1>>>(addr, val++, chain);
      cudaDeviceSynchronize();
      
      cudaMemcpy(host, chunk, CHUNK_SIZE, cudaMemcpyDeviceToHost);
      cudaDeviceSynchronize();
      
      std::cout << "lo: " << lower << " mid: " << mid << " up: " << upper << std::endl;
      
      if (host[first_idx] == NONEVICTED) {
        if (lower == upper) {
          res_set.insert(mid);
          lower = START_POS;
          upper = mid - 1;
          std::cout << "\tfind: " << mid << std::endl;
        } else {
          lower = mid;
        }
      } else {
        upper = mid - 1;
      }
      
      if (res_set.size() > 16) {
        res_set.clear();
        break;
      }
      // A delay (>= 1s) needs to be used! Otherwise, the GPU sometimes fails.
      sleep(1);
    }
    
    res_file << target << "\t";
    for (auto blk : res_set)
      res_file << blk << " ";
    res_file << std::endl;
    res_file.flush();
  }
  
  res_file.close();
  
  delete[] host;
  cudaFree(chunk);
  cudaFree(pad);
}

