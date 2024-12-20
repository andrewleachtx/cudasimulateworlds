# CUDA World Simulations
The goal of this project is to optimize the simulation of $k$ parallel worlds with colliding particle checks (no optimizations; $n^2$ so $n \approx 64$) with each world being a ran as a kernel of its own.

Simulation states are not rendered; you can output position data for the $ith$ world if passed to the program.

## Contents
- [CUDA World Simulations](#cuda-world-simulations)
  - [Contents](#contents)
  - [Dependencies \& Building](#dependencies--building)
  - [Helpful Commands](#helpful-commands)
  - [GPU Specifications](#gpu-specifications)
  - [Notes](#notes)
  - [Logging](#logging)
  - [Optimizations \& New Features](#optimizations--new-features)
  - [Plots \& Images](#plots--images)
  - [Obstacles](#obstacles)
  - [Conclusions](#conclusions)
  - [Future Improvements](#future-improvements)

## Dependencies & Building
1. CUDA `12.x` is **necessary** for this code. It is recommended to use `12.6`, which you can download on the NVIDIA Toolkit website. You can add the cuda version to your path with
    ```sh
    export PATH=/usr/local/cuda-11.7/bin$PATH
    ```
    to `~/.profile` and `source ~/.bashrc`. To know it works, run `nvcc --version`.

2. Install [glm](https://github.com/g-truc/glm/tree/master) somewhere - I added an environment variable `export GLM_INCLUDE_DIR=~/packages/glm` to my `.profile`. You can also just modify `CMakeLists.txt` to point to your glm installation if it's stored local to the project or elsewhere.
3. Run `cmake -B build -DCMAKE_BUILD_TYPE=Release` from project root, then run `cmake --build build`.
4. To execute, you can play with `test/run.sh` (run in `test/`) or run `build/CUDASIMULATEWORLDS <num_worlds> [log_world_idx] [output_dir="../test/results/simdata"]`. To view world and state output for a world, you can add the optional arguments specified above. Also, to control the interval of the csv (slows down execution) you can go to `src/constants.h` and change `WORLD_LOG_STEPSIZE`.

## Helpful Commands
- `cat /etc/os-release` shows the distro and device architecture.
- `nvidia-smi` (append `-l <second interval>` for repeated updates) basic GPU and process information.
- `ncu --target-processes all -o <report name> <executable> [executable args]` will generate a compute report over all kernels given some executable and its args.
- `"...\NVIDIA GPU Computing Toolkit\CUDA\v12.6\extras\demo_suite\deviceQuery"` (or wherever it is located) provides fine-grained information and details about the GPU and its processing power.

## GPU Specifications
Relevant specs for GPUs used in development - the 4090 is on a Linux server, while the A2000 and 2080 SUPER were used locally. Device information was queried with `deviceQuery`.

| Query                       | NVIDIA GeForce RTX 4090          | NVIDIA RTX A2000 Laptop GPU       |
|-----------------------------|----------------------------------|-----------------------------------|
| **CUDA Driver Version**     | 12.6                             | 12.6                              |
| **CUDA Compute Capability** | 8.9                              | 8.6                               |
| **CUDA Cores**              | 16384                            | 2560                              |
| **Total Global Memory**     | 24111 MiB (25282.28 MB)          | 4096 MiB (4294.51 MB)             |
| **L2 Cache Size**           | 75497472 bytes (75.5 MiB)        | 2097152 bytes (2 MiB)             |
| **Registers per Block**     | 65536                            | 65536                             |
| **Constant Memory**         | 65536 bytes                      | 65536 bytes                       |
| **Shared Memory per Block** | 49152 bytes                      | 49152 bytes                       |

## Notes
1. Benchmarks were run on a **NVIDIA GeForce RTX 4090** with **CUDA 12.2**. For better access to the hardware and profiling, a RTX 2080 SUPER was used locally to debug and run against `ncu` to evaluate performance.
   1. The hope of using ncu to isolate inefficiencies is that they will scale to the benchmarks on the full-fledged 4090 GPU or standard.
2. Each simulation has kernel execution timing averaged over a maximum of 10000 or `MAX_STEPS` steps.
3. `test/...` is where many of the .sh scripts were made to iterate over various particle sizes and thread counts.
   1. `stdout` is saved to `test/results/stdout/out_{world}.txt/`.
   2. Simulation data (if toggled) in `test/results/simdata/{world}_{minute}_{second}`.
      1. System time of `minute` and `second` the script was run.
4. GPU Memory:
   1. Global
      1. $kn$ particles are allocated into global memory. Each particle has a unique position $\vec{x_i}$, velocity $\vec{v_i}$ and radius $r_i$, which are stored as `vec4` and `float`, respectively. $$2 * sizeof(vec4) + sizeof(float) = 36 \text{ bytes per particle}$$ $$\implies kn \leq \frac{\text{max bytes}}{\text{sizeof(particle)}} = \frac{25282281472 \text{ bytes}}{36 \text{ bytes}} \approx 702,285,596$$ this is relevant because it means with $n = 64$, we can maximally allocate $\frac{702,285,596}{64} \approx 10973212$ **worlds.** Of course, this value is decreased greatly as not all of the theoretical maximum VRAM is usable for us. If you wanted to exceed this, you could invite CPU fallback memory with **unified memory** at the cost of performance - but this is not enabled for us.
      2. For the $kth$ simulation we can access particle$_{ki}$ by designating 1 block to 1 simulation, and 1 thread to one particle. With $k$ blocks and $n$ threads, we have `blocksPerGrid` = $k$ and `threadsPerBlock` = $n$.
         1. It would look like this: `simulateKernel<<<blocksPerGrid=k, threadsPerBlock=n>>>`
         2. `simulationIdx = blockIdx.x`, `particleIdx = threadIdx.x` $\implies \text{particle}_{ki}$ `idx = simulationIdx * particleIdx`.
5. Every instance of `simulateKernel` is one world; one simulation step for that entire world. If we have $k$ worlds and 1 block = 1 world, the bound is $min(k, \text{max GPU blocks})$. Eventually, we are going to actually exceed the number of available blocks on the GPU, as there are `deviceProp.maxGridSize[0]` blocks available to us. **In practice, because the 4090's maxBlocks = 2,147,483,647** and $k \leq 10,973,212$ there is no point in this, but it is supported in the code as it was a concern at first.
   1. This can be resolved by moving towards a batch approach of launching the kernels in our length $k$ loop, which introduces a degree of series execution as our loop runs $\lceil\frac{numWorlds}{\text{maxBlocks}}\rceil$ times.
   2. The batch size is the minimum of the remaining worlds to process and the maximum block size: $$\min(\text{maxBlocks}, \text{numWorlds} - (i * \text{maxBlocks}))$$
   3. Our number of batches, rewritten to use integer division's implicit floor: $$\left\lceil\frac{numWorlds}{maxBlocks}\right\rceil = \left\lfloor\frac{\text{numWorlds} + \text{maxBlocks} - 1}{maxBlocks}\right\rfloor$$
   4. Note we should recall to offset our $\vec{x}[]$, $\vec{v}[]$, and $r[]$ arrays to account for the correct batch. Also, we should store a convergence flag per world, that way we can reason when there is local vs global convergence.

## Logging
1. With no rendering, you can view the simulation state for a given world by adding additional arguments `./<numWorlds> [world idx to log] [output file directory]` with the desired world index in `[0, numWorlds)`.
2. The output file is formatted as a `.csv` with `step,time,particle,x,y,z`. These are in `rest/results/simdata`

## Optimizations & New Features
1. Moved to explicitly using `glm::vec4` with padding over `glm::vec3`. This is to use 16 bytes per instance. Originally, casts to `glm::vec3` were made, but this seemed a bit redundant and made new allocations - refactored to just ignore the w term.
2. Introduction of shared memory (`__shared__`) for storing per-block data on the GPU, making extremely fast access at the cost of space.
   1. Simulation "convergence" was once decided by the `&&` of all each particle's $\vec{v} = 0$. This would normally be a race condition, but we can use `atomicAnd` to get all of them at once efficiently and safely.
   2. To gracefully handle particle collisions, we can store $\Delta \vec{v_{ki}}\left[n\right]$, and as $n$ is constant, we can do this at compile time (otherwise we could use `extern __shared__ glm::vec3 s_dv[]` and update device properties).
   3. We have `deviceProp.sharedMemPerBlockbytes` of shared memory available to us per block - because this is such a small region (48 KB on 4090) we are using a `vec3` regardless of padding / time concerns. Then we have $\frac{\text{shared memory}}{sizeof(vec3)}$ particles at a max. Assuming 48 KB, that would be $\frac{48 * 2^{10}}{12} = 4096$ particles. Well over enough for our fixed amount $n = 64$.
3. The `ncu` (Nsight Compute) results I addressed are:
   1. Uncoalesced Global Accesses (est. 57.67% speedup)
      1. 68% of total sectors were excessive, meaning for each chunk of data I globally requested, I was only using 32% of it.
      2. The bulk of this comes from 
   2. Uncoalesced Shared Accesses (est. 71.79%) speedup
      1. Added shared memory for position, velocity, and radii to simulateKernel to reduce repeated global access in the `solveConstraints` and its nested loop. This also improved `getAcceleration` negligibly. Additionally this reduced some register usage as I could reuse shared memory instead of making temporary variables.
   3. Warp Divergence and Stalling
      1. It is clear from the **Warp State Statistics** tab that many of my warps are stalling, which I believe to be caused by a wait after synchronization, as well as control flow altering warp control flow.
      2. Restructured nested loop in `solveConstraints` from an inner loop of `j = particleIdx + 1, j < d_numParticles` to `j = 0; j < d_numParticles` with an `if (particleIdx < j)` condition. Ultimately there will always be stalls due to the usage of shared memory as well as the `atomic` calls.
   4. High Local Memory Utilization
      1. Register spills result in memory being sent to global from a warp, and I used quite a few helper variables that on a more massive scale actually resulted in 74.96% local memory usage, which is not good.
      2. I removed many of the local variables, reusing direct access to shared memory array access.
4. `getAcceleration`
   1. Moved to `__inline__`, removed unnecessary allocations, and removed unused air resistance term.

## Plots & Images
<!-- ![Total Kernel Time x World Count](./figures/totalKernelTime_x_worldCt.png) -->
<!-- ![Global Memory (bytes) x World Count](./figures/globalMemory_x_worldCt.png) -->
<p style="text-align: center">
    <img src="./figures/totalKernelTime_x_worldCt.png" width=75% alt="Total Kernel Time x World Count">
    <img src="./figures/globalMemory_x_worldCt.png" width=75% alt="Global Memory (bytes) x World Count">
</p>

## Obstacles
1. While official benchmarks were run on the GeForce RTX 4090, for development a local GPU(s) was used to debug and run against `ncu` to evaluate performance.
   1. This is a result of complications with using the VM without root/administrator privileges. After working with administrators of the machine, sufficient permissions to execute `ncu` were given, but then new errors came in trying to profile the GPU.
   2. The hope of using ncu to isolate inefficiencies is that they will scale to the benchmarks on the full-fledged 4090 GPU or standard.
2. Somehow, linkage to critical CUDA files was lost on the Linux machine, including `/usr/local/cuda`. There was a lot of files that had to be deleted. Much time was lost having to simply debugging and fixing compilation due to a standard library `std_function.h` function syntax.
   1. Relinked my path to account for the missing `/usr/local/cuda` directory by finding and relinking to an existing CUDA version on the machine.
   2. Added necessary lines to CMakeLists.txt to manually link to the correct version.
   3. The only cuda version I could find was cuda `11.x`, which `nvcc` has known issues compiling `g++-11`. The fix is to update to a cuda `12.x` version, which is what I was using before it vanished, or downgrade to `g++-10` which doesn't contain the `std_function.h` conflict. However, `g++-10` is also not on the system, and without root access I can't install it.
   4. To avoid losing further time awaiting administrator permissions, I locally installed the [CUDA 12.6 Toolkit](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=runfile_local) to my home (~) directory, and linked it there. There are still some failed linkages when building, i.e. `nvlink warning : Skipping incompatible '/lib/x86_64-linux-gnu/librt.a' when searching for -lrt` because the resource is gone, but they are not necessary.

## Conclusions
1. After about $2^{12}$ worlds, you can observe a constant to linear change in average `simulateKernel` time. It was believed that this was due to a hardware limit - if there are `16384` CUDA cores, and we simulate a world per core, we can only achieve parallelization up to `16384` worlds, right?
   1. This was wrong. While there are 16384 CUDA cores, my graph of GPU information is **misleading** in that the GPU does not execute cores on an individual level like this.
   2. The streaming multiprocessors on a 4090 handle about 32 blocks simultaneously, with 128 SMs. This means we have a maximum around $128 \text{SMs} * 32 \text{blocks per SM} = 4096$ blocks available to be parallelized, which explains our $2^{12}$ inflection point.
2. The design of this simulation is inherently flawed.
   1. From the start, the simulation was designed for 1 block per world with `k` worlds with a fixed $n=64$ particles per block. This simplified things, and made intuitive sense; one `simulateKernel` instance was called for each world, and each one had `64` particles or threads.
   2. This is problematic because our block can store `1024` threads or particles, and I only used `64`. I think this quick sketch helps: <p align="center"><img src="./figures/gpu_hierarchy.png" width=60% /></p>
   3. As you can see, I am only utilizing $2 / 32 = 6.25%$ of warps available per block. Warps execute in rapid series per block - what does this mean for us?
      1. Occupancy (ratio of active to available warps) drops massively, and so does latency hiding.
         1. When one of the worlds (2 warps) idles, we do not have other warps available to be swapped in to "hide" the latency of waiting.
      2. Even if we increase the number of worlds per block, at the same world count as before, wouldn't this only slow us down as there are technically 32 warps or 16 worlds executing in rapid series?
         1. Think of increasing warps as increasing the amount of warps the SM can work with to continuously have something to work with; it is a good thing.
         2. Don't forget by doing this you greatly reduce the number of blocks, and that can 1) reduce scheduling and joining overhead, 2) better use shared memory / registers, etc... 

## Future Improvements
1. Formatted stdout + Python parsing as a system for outputting, retrieving, and analyzing benchmark data could be vastly improved if not changed.
   1. For example, printing "`[BENCHMARK] Average individual simulateKernel() time over 6609 samples: 0.027232 ms`" and could be rewritten more concisely to ease regex for text parsing to retrieve what I need.
   2. Use `csv` for more standard data intake w/ Python?
2. These `README` files are too long.
   1. Specifically allocating time a **day** after writing them to trim fat would be helpful.
3. The plots are a black box and hectic to generate - `matplotlib` is used.
   1. ~~Create a general template or plot style?~~
4. There needs to be a way to output the `README.md` as a pdf, as currently doing so works, but breaks `LaTeX` which I love using for math too much to remove.
5. While world output logging exists, it is unknown how to read it somewhere.