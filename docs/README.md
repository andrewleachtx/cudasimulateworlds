# CUDA World Simulations
The goal of this project is to run $k$ parallel worlds simulating particles with collision checks (no optimizations; $n^2$ so $n \approx 64$) with each world being a ran as a kernel of its own.

Simulation states are not rendered; you can output the $ith$ world simulation state data if you specify to.

## Dependencies (Linux)
1. To get the VM CUDA to run, append
```sh
export PATH=/usr/local/cuda-11.7/bin$PATH
```
to `~/.profile` and `source ~/.bashrc`. To know it works, run `nvcc --version`.

2. Install GLM somewhere - I added an environment variable `export GLM_INCLUDE_DIR=~/packages/glm` to my `.profile` and got the code from [here](https://github.com/g-truc/glm/tree/master). Feel free to just modify `CMakeLists.txt` to point to your glm installation if its local to the project, or elsewhere.

## Helpful Commands
- `cat /etc/os-release` shows the distro and device architecture.
- `nvidia-smi` provides basic GPU information (assuming it is a NVIDIA gpu).
- `ncu --target-processes all -o <report name> <executable> [executable args]` will generate a compute report over all kernels given some executable and its args.
- `"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\extras\demo_suite\deviceQuery.exe"` (or wherever it is located) will provide even more specific device information.

## GPU Specifications
1. TODO: Insert a chart differentiating the 4090, 2080 SUPER, and A2000 using deviceQuery.exe 

## Notes
1. Benchmarks were run on a **NVIDIA GeForce RTX 4090** with **CUDA 12.2**. For better access to the hardware and profiling, a RTX 2080 SUPER was used locally to debug and run against `ncu` to evaluate performance.
   1. The hope of using ncu to isolate inefficiencies is that they will scale to the benchmarks on the full-fledged 4090 GPU or standard.
   2. 
2. Each simulation "converges" when their last particle velocity reaches $\approx 0$. For the $ith$ world or simulation, this occurs when the "dead particle" counter reaches the size $n_i$ of that simulation.
3. Thread size and particle size vary and are tested as hyperparameters. Block size is according to the following:
   ```cpp
    size_t problem_sz = g_particles.h_maxParticles;
    g_blocksPerGrid = dim3((problem_sz + g_threadsPerBlock.x - 1) / g_threadsPerBlock.x);
    ```
4. `test/...` is where many of the .sh scripts were made to iterate over various particle sizes and thread counts.
   1. The files are labeled in `cout/{particle ct}/`.
5. GPU Memory:
   1. Global
      1. $kn$ particles are allocated into global memory. Each particle has a unique position $\vec{x_i}$, velocity $\vec{v_i}$ and radius $r_i$, which are stored as `vec4` and `float`, respectively. $$2 * sizeof(vec4) + sizeof(float) = 36 \text{ bytes per particle}$$
      2. For the $kth$ simulation we can access particle$_{ki}$ by designating 1 block to 1 simulation, and 1 thread to one particle. With $k$ blocks and $n$ threads, we have `blocksPerGrid` = $k$ and `threadsPerBlock` = $n$.
         1. It would look like this: `simulateKernel<<<blocksPerGrid=k, threadsPerBlock=n>>>`
         2. `simulationIdx = blockIdx.x`, `particleIdx = threadIdx.x` $\implies \text{particle}_{ki}$ `idx = simulationIdx * particleIdx`.
6. Every instance of `simulateKernel` is one world; one simulation step for that entire world. If we have $k$ worlds and 1 block = 1 world, the bound is $min(k, \text{max GPU blocks})$. Eventually, we are going to actually exceed the number of available blocks on the GPU, as there are `deviceProp.maxGridSize[0]` blocks available to us.
   1. This can be resolved by moving towards a batch approach of launching the kernels in our length $k$ loop, which introduces a degree of series execution as our loop runs $\lceil\frac{numWorlds}{\text{maxBlocks}}\rceil$ times.
   2. The batch size is the minimum of the remaining worlds to process and the maximum block size: $$\min(\text{maxBlocks}, \text{numWorlds} - (i * \text{maxBlocks}))$$
   3. Our number of batches, rewritten to use integer division's implicit floor: $$\left\lceil\frac{numWorlds}{maxBlocks}\right\rceil = \left\lfloor\frac{\text{numWorlds} + \text{maxBlocks} - 1}{maxBlocks}\right\rfloor$$
   4. Note we should recall to offset our $\vec{x}[]$, $\vec{v}[]$, and $r[]$ arrays to account for the correct batch. Also, we should store a convergence flag per world, that way we can reason when there is local vs global convergence.

## Logging
1. With no rendering, you can view the simulation state for a given world by adding additional arguments `./<numWorlds> [world idx to log] [output file directory]` with the desired world index in `[0, numWorlds)`.
2. The output file is formatted in a w

## Optimizations & New Features
1. Moved to explicitly using `glm::vec4` with padding over `glm::vec3`. This is to use 16 bytes per call.
2. Introduction of shared memory (`__shared__`) for storing per-block data on the GPU, making extremely fast access at the cost of space.
   1. Simulation "convergence" is decided by the `&&` of all each particle's $\vec{v} = 0$. This would normally be a race condition, but we can use `atomicAnd` to get all of them at once efficiently and safely.
   2. To gracefully handle particle collisions, we can store $\Delta \vec{v_{ki}}\left[n\right]$, and as $n$ is constant, we can do this at compile time (otherwise we could use `extern __shared__ glm::vec3 s_dv[]` and update device properties).
   3. We have `deviceProp.sharedMemPerBlockbytes` of shared memory available to us per block - because this is such a small region (48 KB on 4090) we are using a `vec3` regardless of padding / time concerns. Then we have $\frac{\text{shared memory}}{sizeof(vec3)}$ particles at a max. Assuming 48 KB, that would be $\frac{48 * 2^{10}}{12} = 4096$ particles. Well over enough for our fixed amount $n = 64$.
3. The `ncu` (Nsight Compute) results I addressed based on 256 worlds using a 2080 SUPER are:
   1. Uncoalesced Global Accesses (est. 57.67% speedup)
      1. 68% of total sectors were excessive, meaning for each chunk of data I globally requested, I was only using 32% of it.
      2. The bulk of this comes from 
   2. Uncoalesced Shared Accesses (est. 71.79%) speedup
      1. tbd
   3. Warp Divergence 
      1. It is clear from the **Warp State Statistics** tab that many of my warps are stalling, which I believe to be caused by a wait after synchronization.
      2. The line accounting for the most stall sampling is the header of this for loop: `for (int i = 0; i < d_numPlanes; i++) {`
 

## Plots & Images

## Obstacles
1. While official benchmarks were run on the GeForce RTX 4090, for development a local GPU(s) was used to debug and run against `ncu` to evaluate performance.
   1. The hope of using ncu to isolate inefficiencies is that they will scale to the benchmarks on the full-fledged 4090 GPU or standard. 