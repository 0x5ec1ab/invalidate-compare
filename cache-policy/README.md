These experiments are performed on **Ubuntu 20.04** with NVIDIA driver version 515.76.

---

First, *to reduce interference during experiments, we use the integrated GPU for display*.
Therefore, we set the BIOS to use the CPU-integrated GPU instead of the discrete one.
When you use `nvidia-smi`, you should see that **no processes are using the discrete GPU**. 

---

Build all the experiments:

```
make
```

- Check whether L2 is inclusive or non-inclusive: 
  ```
  ./inclusion-policy
  ```
  
  We mainly focus on SM Y's new value:
  - If L2 is inclusive, the value should be `0`. 
  - If L2 is non-inclusive, the value should be `0xbeefbeef`.
  
  It should give you the following results:
  ```
  SM X: old value 0 	 new value 0
  SM Y: old value 0 	 new value beefbeef
  ``` 
  
  This means L2 is non-inclusive.
  
  
- Check L1's write policy (write-back or write-through):
  ```
  ./write-policy
  ```
  
  We mainly focus on SM Y's value:
  - If L1 uses write-back, the value should be `0`.
  - If L1 uses write-through, the value should be `0xdeadbeef`.
  
  It should give you the following results:
  ```
  SM Y: value deadbeef
  ```
  
  This means L1 uses write-through.
  
  
- Check L1's write-allocate policy (write-allocate or no-write-allocate):
  ```
  ./l1-allocate-policy
  ```
  
  We mainly focus on SM X's value:
  - If L1 uses write-allocate (i.e., cache line is allocated on a write miss), the value should be `0xdeadbeef`.
  - If L1 uses no-write-allocate (i.e., write misses do not affect the cache), the value should be `0`.
  
  It should give you the following results:
  ```
  SM X: value deadbeef
  ```
  
  This means L1 uses write-allocate.
  
  
- Check L2's write-allocate policy (write-allocate or no-write-allocate):
  ```
  ./l2-allocate-policy
  ```
  
  We mainly focus on SM X's new value:
  - If L2 uses write-allocate (i.e., cache line is allocated on a write miss), the 2nd value should be the same as the 1st one.
  - If L2 uses no-write-allocate (i.e., write misses do not affect the cache), the 2nd value should be `0xdeadbeef`.
  
  It should give you the following results:
  ```
  SM X: 1st value 0
  SM X: 2nd value 0
  ```
  
  This means L2 uses write allocate.
  
  
- Verify L1's autoflush behavior:
  
  Run `dummy` first to create a GPU context. It is just a self-looping branch instruction.
  ```
  ./dummy
  ```
  After `dummy` is running, we run:
  ```
  ./l1-autoflush
  ```
  
  We check SM Y's value:
  - If it is `0`, L1 has no autoflush behavior.
  - If it is `0xdeadbeef`, L1 has the described autoflush behavior.
  
  It should give you the following results:
  ```
  SM Y: value deadbeef
  ```

  This verifies L1's autoflush behavior.


