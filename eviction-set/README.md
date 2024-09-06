We tested this eviction set finder on **Ubuntu 20.04** with NVIDIA driver version 515.76.

---

First, *to reduce interference during eviction set finding, we use the integrated GPU for display*. Therefore, we set the BIOS to use the CPU-integrated GPU instead of the discrete one. When you use `nvidia-smi`, you should see that **no processes are using the discrete GPU**. If you still see `/usr/lib/xorg/Xorg` and/or `/usr/bin/gnome-shell`, try adding a blacklist file to blacklist the `nvidia_drm` module. 

---

Build the finder:
```
make
```

---

Run the finder:
```
./finder
```

It will report `chunk`'s virtual address. To find `chunk`'s physical GPU memory address, you can use our [GPU memory dumper and page table extractor](https://github.com/0x5ec1ab/gpu-tlb). 

Modify `PAD_SIZE` to ensure the physical address of `chunk` starts at your desired address in GPU memory (e.g., 0x20000000). You also need to adjust the number of memory blocks `TARGET_NUM` to reflect the number of cache sets for which you want to find eviction sets. After these changes, **rebuid the finder**.

---

Run the finder (provide a file name which will be used to save the found eviction sets):
```
./finder <file to save eviction sets>
```

---

Note that for ***consumer-grade GPUs (e.g., RTX 3080)***, this finder program yields **7 memory blocks** for a target cache set. To obtain all 16 memory blocks, adjust the `START_POS` to a larger value. For ***server-grade GPUs (e.g., A10)***, this finder program can once yield **15 memory blocks** for a target cache set. However, this does **not** mean the dirty cache line ratio is greater than 50%. In fact, there are still only 7 dirty blocks at any given time. When the program runs beyond 7 memory blocks, it starts evicting the oldest of the 7 dirty blocks. This process continues for the next 8 blocks, resulting in a total of 15 yielded blocks, but with only 7 being dirty at any moment.

---
***WE WILL ADD SOME POST-PROCESSING SCRIPTS SOON***


