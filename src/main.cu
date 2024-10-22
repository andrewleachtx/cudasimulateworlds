#include "include.h"

#include "constants.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <chrono>
#include <iomanip>
#include <fstream>
#include <sstream>

using std::cout, std::cerr, std::endl;
using std::vector, std::string, std::make_shared, std::shared_ptr;
using std::stoi, std::stoul, std::min, std::max, std::numeric_limits, std::abs;

int g_worldLogIdx = -1;
string g_worldLogOutDir = "";
std::ofstream g_worldLogStream;

static const size_t g_numParticles = NUM_PARTICLES;
static size_t g_numWorlds, g_maxBlocks;
static int* h_convergenceFlags;
static bool g_isGlobalConverged(false);
std::chrono::high_resolution_clock::time_point g_progStart;
float g_curStepTime(0.0f);
long long g_curStep(0);

// Device Hyperparameters - Constant Space //
__constant__ size_t d_numParticles;
__constant__ size_t d_numWorlds, d_numPlanes;
__constant__ glm::vec4 d_planeP[6], d_planeN[6];

bool g_is_simFrozen(false);
cudaEvent_t cudaEvt_simStart, cudaEvt_simStop;

// blocks = k, threads = n
dim3 g_blocksPerGrid;
dim3 g_threadsPerBlock;

// static const int g_timeSampleSz = KERNEL_TIMING_SAMPLESZ;
static size_t g_timeSampleCt = 0;
static float g_totalKernelTime(0.0f), g_totalBatchLoopTime(0.0f);

ParticleData g_particles;
PlaneData g_planes;

static void init() {
    srand(0);
    size_t k = g_numWorlds;

    // CUDA //
        gpuErrchk(cudaSetDevice(0));
        cudaEventCreate(&cudaEvt_simStart);
        cudaEventCreate(&cudaEvt_simStop);

    // Planes //
        const float plane_width = 540.0f;
        g_planes = PlaneData(6, plane_width);
        g_planes.initPlanes();
        g_planes.copyToDevice();

    // Particles //
        g_particles = ParticleData(g_numParticles, k);
        g_particles.init(0.5f);
        g_particles.copyToDevice();

        // numWorlds
        gpuErrchk(cudaMemcpyToSymbol(d_numWorlds, &g_numWorlds, sizeof(size_t)));

        // We should zero h_convergenceFlags here and send that to CUDA
        h_convergenceFlags = new int[k];
        memset(h_convergenceFlags, 0, k * sizeof(int));
        cudaMemcpy(g_particles.d_convergenceFlags, h_convergenceFlags, k * sizeof(int), cudaMemcpyHostToDevice);
}

/*
    Instead of iterating over each particle, we will make a kernel that runs for each particle
*/

__device__ __inline__ glm::vec4 getAcceleration() {
    // Simple force composed of gravity - you can add air resistance term (or others) and pass in velocity.
    glm::vec4 F_total = glm::vec4(0.0f, GRAVITY, 0.0f, 0.0f);

    return F_total;
}

/*
    In a flocking simulation, you might have various rules and applications to follow - in this case, we are just
    going to establish a simple distance constraint that is resolve with impulse / momentum.
*/

static __device__ void solveConstraints(int idx, const glm::vec4* s_pos, const glm::vec4* s_vel, const float* s_radii,
                                 glm::vec4& x_new, glm::vec4& v_new, float& dt, const glm::vec4& a,
                                 int simulationIdx, int particleIdx, glm::vec3* s_dv, int& is_converged) {
    // We could truncate
    glm::vec4 x(s_pos[particleIdx]), v(s_vel[particleIdx]);
    const float r_i = s_radii[particleIdx];

    // Particle-Particle Collisions //
    /*
        This could be below plane collisions, but seeing as we synchronized threads, we will do it here

        Because of warp divergence, I have changed from an inner loop of j = particleIdx + 1, j < d_numParticles
        to j = 0; j < d_numParticles. It is possible to do a pairwise logic, but ultimately warps will have to
        wait no matter what, because the fewer collision checks one warp has to do.

        This code has been optimized to reduce local memory usage, so look at old commits for logic
    */

    for (int j = 0; j < d_numParticles; j++) {
        if (particleIdx < j) {
            int idx_j = simulationIdx * d_numParticles + j;

            glm::vec4 x_ij = x - s_pos[j];

            // FIXME: Length calls like this can be dangerous if the w component isn't 0.0f
            float d_ij = glm::length(x_ij);

            if (d_ij < r_i + s_radii[j]) {
                glm::vec4 n_ij = glm::normalize(x_ij);

                float impulse = (1 + RESTITUTION) * glm::dot(v - s_vel[j], n_ij) / (1 + 1);

                glm::vec4 impulse_vec = impulse * n_ij;
                s_dv[particleIdx] += glm::vec3(impulse_vec);

                // Consequently, we should change the velocities of j. This is not thread safe so we have to atomic
                atomicAdd(&s_dv[j].x, -impulse_vec.x);
                atomicAdd(&s_dv[j].y, -impulse_vec.y);
                atomicAdd(&s_dv[j].z, -impulse_vec.z);
            }
        }
    }

    // Synchronize threads and add impulse results - we have to do this before the plane collisions
    __syncthreads();
    v += glm::vec4(s_dv[particleIdx], 0.0f);

    // Plane Collisions with x, v = glm::vec4 now //
    for (int i = 0; i < d_numPlanes; i++) {
        const glm::vec4 n(d_planeN[i]);
        const glm::vec4 p_offset(d_planeP[i] + (r_i * n));

        float d_0 = glm::dot(x - p_offset, n);
        float d_n = glm::dot(x_new - p_offset, n);

        glm::vec4 v_tan = v - (glm::dot(v, n) * n);
        v_tan = (1 - FRICTION) * v_tan;

        if (d_n < FLOAT_EPS) {
            float f = d_0 / (d_0 - d_n);
            dt = f * dt;

            glm::vec4 v_collision = (v + (dt * a)) * RESTITUTION;    
            glm::vec4 x_collision = x;

            x_new = x_collision;
            v_new = (abs(glm::dot(v_collision, n)) * n) + (v_tan);
        }

        // Convergence or jitter check - (v = 0, "on" the plane, and acceleration towards plane)
        if ((length(v_new) < STOP_VELOCITY) && (d_n < 0.1f) && (dot(a, n) < FLOAT_EPS)) {
            v_new = glm::vec4(0.0f);
            is_converged = 1;
        } 
    }
}

/*
    We are now working with global arrays of great size; 
*/
__global__ void simulateKernel(glm::vec4* pos, glm::vec4* vel, float* radii, int* convergeFlags) {
    unsigned int simulationIdx(blockIdx.x), particleIdx(threadIdx.x);

    // Overflow shouldn't be possible
    int idx = simulationIdx * d_numParticles + particleIdx;

    // FIXME: When a thread returns early, it cannot join __syncthreads later, so we should note this
    if (idx >= (d_numWorlds * d_numParticles)) {
        printf("[ERROR] Returning idx = %d\n", idx);        
        return;
    }

    // Allocate shared memory for graceful impulse & convergence handling; each block is one world so this works 
    __shared__ glm::vec3 s_dv[NUM_PARTICLES];
    __shared__ int s_converged;

    // Shared memory for position, velocity, and radii to reduce repeated global access
    __shared__ glm::vec4 s_pos[NUM_PARTICLES];
    __shared__ glm::vec4 s_vel[NUM_PARTICLES];
    __shared__ float s_radii[NUM_PARTICLES];

    // Each thread should then load this, now we have much closer memory to our warp
    s_pos[particleIdx] = pos[idx];
    s_vel[particleIdx] = vel[idx];
    s_radii[particleIdx] = radii[idx];

    __syncthreads();

    // We only want to initialize it once
    if (particleIdx == 0) {
        s_converged = 1;
    }

    // Handle fractional timesteps
    float dt_remaining = DT_SIMULATION;
    float dt = dt_remaining;
    short max_iter = 10;
    int is_stopped = 0;
    
    while (max_iter && dt_remaining > FLOAT_EPS) {
        // Within the timestep multiple collisions are possible, so we will have to reuse the shared memory 
        s_dv[particleIdx] = glm::vec3(0.0f);
        
        glm::vec4 a = getAcceleration();

        // Integrate over timestep to update
        glm::vec4 x_new = s_pos[particleIdx] + (dt * s_vel[particleIdx]);
        glm::vec4 v_new = s_vel[particleIdx] + (dt * a);

        // We have to synchronize before and after entering
        __syncthreads();

        // Resolve particle-particle AND particle-plane position constraints - should only use shared memory now
        solveConstraints(idx, s_pos, s_vel, s_radii, x_new, v_new, dt, a, simulationIdx, particleIdx, s_dv, is_stopped);

        __syncthreads();

        s_pos[particleIdx] = x_new;
        s_vel[particleIdx] = v_new;

        dt_remaining -= dt;
        max_iter--;
    }

    // The and of all 64 particles being stopped in this world being 1 represents full convergence
    atomicAnd(&s_converged, is_stopped);

    // No need for atomic here, only the first thread will update the flag
    if (particleIdx == 0) {
        convergeFlags[simulationIdx] = s_converged;
    }
    
    // At the end update the global memory
    pos[idx] = s_pos[particleIdx];
    vel[idx] = s_vel[particleIdx];
}

void launchSimulations(std::ostream& output_buf, glm::vec4* pos_buf, vector<float>& h_worldConvergenceTimes) {
    size_t maxBlocks(g_maxBlocks), numWorlds(g_numWorlds);
    int batch_ct = (numWorlds + maxBlocks - 1) / maxBlocks;

    auto t_batchLoopStart = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < batch_ct; i++) {
        int batch_offset = i * maxBlocks;
        int batch_sz = std::min(maxBlocks, numWorlds - batch_offset);

        // We should offset our pointers correspond to the correct batch
        glm::vec4* pos = g_particles.d_position + (batch_offset * g_numParticles);
        glm::vec4* vel = g_particles.d_velocity + (batch_offset * g_numParticles);
        float* radii = g_particles.d_radii + (batch_offset * g_numParticles);
        int* c_flags = g_particles.d_convergenceFlags + (batch_offset);

        // If specified, we will output a specific world's position data over time for each particle
        if ((g_curStep % 50 == 0) && g_worldLogIdx != -1 && g_worldLogIdx >= batch_offset && g_worldLogIdx < batch_offset + batch_sz) {
            int world_offset = (g_worldLogIdx - batch_offset) * g_numParticles;
            cudaMemcpy(pos_buf, pos + world_offset, g_numParticles * sizeof(glm::vec4), cudaMemcpyDeviceToHost);

            // csv format of |cur_step|cur_time|particle|x|y|z|
            for (int p = 0; p < g_numParticles; p++) {
                output_buf << g_curStep << "," << g_curStepTime << "," << p << "," << pos_buf[p].x << "," << pos_buf[p].y << "," << pos_buf[p].z << '\n';
            }
        }

        // Launch kernel, static size shared memory should be 64 * sizeof(glm::vec3) ~ 700 bytes per block should be ok
        // https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/#static_shared_memory
        gpuErrchk(cudaEventRecord(cudaEvt_simStart));
        simulateKernel<<<batch_sz, g_threadsPerBlock>>>(pos, vel, radii, c_flags);
        gpuErrchk(cudaEventRecord(cudaEvt_simStop));
        gpuErrchk(cudaEventSynchronize(cudaEvt_simStop));

        // FIXME: Do we need to sync here? 
        // gpuErrchk(cudaDeviceSynchronize());
        gpuErrchk(cudaGetLastError());
    }
    auto t_batchLoopStop = std::chrono::high_resolution_clock::now();

    // Global Convergence //
    bool is_globalConverged = true;
    cudaMemcpy(h_convergenceFlags, g_particles.d_convergenceFlags, numWorlds * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < numWorlds; i++) {
        is_globalConverged = is_globalConverged && h_convergenceFlags[i];
        
        if (BENCHMARK && h_convergenceFlags[i] && h_worldConvergenceTimes[i] < 0.0f) {
            auto conv_time = std::chrono::high_resolution_clock::now() - g_progStart;
            float conv_time_ms = std::chrono::duration<float, std::milli>(conv_time).count();
            
            h_worldConvergenceTimes[i] = conv_time_ms;
        }
    }

    // Could just set it equal, but this way we avoid global access :)
    if (is_globalConverged) {
        g_isGlobalConverged = true;
    }

    // Benchmarking //
    if (BENCHMARK) {
        float t_kernel;
        cudaEventElapsedTime(&t_kernel, cudaEvt_simStart, cudaEvt_simStop);
        g_totalKernelTime += t_kernel;

        float t_batchLoopTime = std::chrono::duration<float, std::milli>(t_batchLoopStop - t_batchLoopStart).count();
        g_totalBatchLoopTime += t_batchLoopTime;

        g_timeSampleCt++;
    }
}

int main(int argc, char**argv) {
    if (argc < 2) {
        cout << "Usage: ./executable <number of worlds/blocks> [world idx to log] [output file directory=../test/results/simdata] " << endl;
        return 0;
    }

    g_numWorlds = (size_t)std::stoull(argv[1]);
    if (g_numWorlds <= 0) {
        cerr << "Number of worlds must be > 0" << endl;
        return 1;
    }

    // Assuming world index AND output directory are given, then we will view 
    glm::vec4* pos_buf = nullptr;
    if (argc >= 3) {
        g_worldLogIdx = std::stoi(argv[2]);
        
        if (argc == 3) {
            g_worldLogOutDir = "../test/results/simdata/";
        }
        else {
            g_worldLogOutDir = string(argv[3]);
        }
        pos_buf = new glm::vec4[g_numParticles];

        if (g_worldLogIdx >= g_numWorlds) {
            cerr << "World log index must be in [0, numWorlds)!" << endl;
            return 1;
        }

        // If missing '/' don't exit, just add it
        if (g_worldLogOutDir[g_worldLogOutDir.size() - 1] != '/') {
            g_worldLogOutDir += "/";
        }

        cout << "[ALOG] Writing state output to " << g_worldLogOutDir << " for world " << g_worldLogIdx << endl;

        // https://stackoverflow.com/questions/16357999/current-date-and-time-as-string
        auto t = std::time(0);
        auto tm = *std::localtime(&t);

        std::ostringstream oss;
        oss << std::put_time(&tm, "%M_%S");

        string output_fname = "world_" + std::to_string(g_worldLogIdx) + "_" + oss.str() + ".csv";
        g_worldLogStream = std::ofstream(g_worldLogOutDir + output_fname);
        g_worldLogStream << "step,time,particle,x,y,z\n";
    }

    // Get GPU info https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#l2-cache-set-aside-for-persisting-accesses
    cudaDeviceProp deviceProp;
    int device;

    cudaGetDevice(&device);
    cudaGetDeviceProperties(&deviceProp, device);

    printf("[INFO] Max grid sizes per dimension are x = %d, y = %d, z = %d\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
    printf("[INFO] Max threads per block: %zu, max shared memory (b): %zu, L2 cache size (b): %zu, global memory size (b): %zu\n", deviceProp.maxThreadsPerBlock, deviceProp.sharedMemPerBlock, deviceProp.l2CacheSize, deviceProp.totalGlobalMem);
    // There will never be a case where we need more than 10 million blocks as we can only cudaMalloc so much.
    g_maxBlocks = min((size_t)deviceProp.maxGridSize[0], (size_t)(1 << 28) - 1);
    printf("[INFO] Batching in %zu worlds / %zu max blocks\n", g_numWorlds, g_maxBlocks);
    
    g_threadsPerBlock = dim3(g_numParticles);
    printf("[INFO] Setting g_blocksPerGrid = dim3(min(%zu, %zu))\n", g_numWorlds, g_maxBlocks);
    g_blocksPerGrid = dim3(std::min(g_numWorlds, g_maxBlocks));

    // Initialize planes, particles, cuda buffers
    init();

    // Program converges when the last moving particle "stops", or the max time is exceeded.
    g_progStart = std::chrono::high_resolution_clock::now();
    auto end = g_progStart + std::chrono::seconds(MAX_SIMULATE_TIME_SECONDS);

    vector<float> h_worldConvergenceTimes(g_numWorlds, -1.0f);
    while (!g_isGlobalConverged && (std::chrono::high_resolution_clock::now() < end)) {
        launchSimulations(g_worldLogStream, pos_buf, h_worldConvergenceTimes);
        
        g_curStep++;
        g_curStepTime = g_curStep * DT_SIMULATION;
    }
    
    // Convergence time
    auto conv_time = std::chrono::high_resolution_clock::now() - g_progStart;
    auto conv_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(conv_time).count();
    printf("[BENCHMARK] Actual program time: %ld ms\n", conv_time_ms);

    // Print Timings //
    if (BENCHMARK) {
        float overall = g_totalKernelTime;
        float avg = g_totalKernelTime / g_timeSampleCt;
        float usage = g_totalKernelTime / (conv_time_ms);

        printf("[BENCHMARK] threadsPerBlock=particlesPerWorld: %d, blocksPerGrid=numWorlds: %d\n", g_threadsPerBlock.x, g_blocksPerGrid.x);
        printf("[BENCHMARK] Average individual simulateKernel() time over %d samples: %f ms\n", g_timeSampleCt, avg);
        printf("[BENCHMARK] Total time spent in kernel before global convergence: %f ms\n", overall);
        printf("[BENCHMARK] Kernel time / total program time: %f\n", usage);
        printf("[BENCHMARK] ----------------------------------\n");
        printf("[BENCHMARK] Total time sending and executing batches of simulateKernel(): %f ms\n", g_totalBatchLoopTime);
        printf("[BENCHMARK] Average time per batch loop: %f ms\n", g_totalBatchLoopTime / g_timeSampleCt);

        float minConvTime(std::numeric_limits<float>::max()), maxConvTime(0.0f), avgConvTime(0.0f);
        for (int i = 0; i < g_numWorlds; i++) {
            if (h_worldConvergenceTimes[i] < minConvTime) {
                minConvTime = h_worldConvergenceTimes[i];
            }

            if (h_worldConvergenceTimes[i] > maxConvTime) {
                maxConvTime = h_worldConvergenceTimes[i];
            }

            avgConvTime += h_worldConvergenceTimes[i];
        }
        
        avgConvTime /= g_numWorlds;
        printf("[BENCHMARK] (Local) Min convergence time: %f ms, Max convergence time: %f ms, Avg convergence time: %f ms\n", minConvTime, maxConvTime, avgConvTime);
    }

    // Cleanup //
    cudaEventDestroy(cudaEvt_simStart);
    cudaEventDestroy(cudaEvt_simStop);
    delete[] h_convergenceFlags;
    delete[] pos_buf;
    if (g_worldLogIdx != -1) {
        g_worldLogStream.close();
    }

    return 0;
}
