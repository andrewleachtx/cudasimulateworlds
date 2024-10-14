#include "include.h"

#include "constants.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <chrono>

using std::cout, std::cerr, std::endl;
using std::vector, std::string, std::make_shared, std::shared_ptr;
using std::stoi, std::stoul, std::min, std::max, std::numeric_limits, std::abs;

/*
TODO:
    1. Initialization
        - We need g_numWorlds * g_numParticles data
            i) Initialize g_numWorlds instances of ParticleData()
        - We could create a world class and tie the ParticleData* to it
            ie. class world - int particlect, int id = ...
    
    2. Simulate
        for world in worlds:
            simulateKernel(ParticleData*)

        simulateKernel is more difficult, we should 
    
    3. Logging
        - We will need to look into options for this, i.e. we could export
            --- TIME 0 ---
            Particle 0: POS_0 VEL_0
            Particle 1: POS_1 VEL_1
            ...
            Particle n-1: POS_n-1 VEL_n-1

        into results/output

    3. Scaling / Output
        - Should be concise, don't need 1000 plots
        - We need to benchmark 1, 2, 4, 8, 16, 32, 64, 128 worlds
        - We should evaluate convergence time and maybe some other measure?
            i) Could do the min, avg, max of time convergence per worldcount
        - Can provide 1-2 insightful NCU charts and what they meant for optimization

    4. Optimization
        - Use NCU / timings to evaluate the bottlenecks, from there document
          rewrites
        - After code is improved as much as possible, can start plotting

*/

static const size_t g_numParticles = NUM_PARTICLES;
static size_t g_numSimulations, g_numParticles;
float g_curTime(0.0f);
long long g_curStep(0);

// Device Hyperparameters - Constant Space //
__constant__ const size_t d_numParticles = NUM_PARTICLES; 
__constant__ size_t d_numWorlds, d_numPlanes;
__constant__ glm::vec4 d_planeP[6], d_planeN[6]; // TODO: FIX SHAPE.CPP IMPLICATIONS OF USING VEC4

bool g_is_simFrozen(false);
cudaEvent_t kernel_simStart, kernel_simStop;

// blocks = k, threads = n
dim3 g_blocksPerGrid;
dim3 g_threadsPerBlock;

// static const int g_timeSampleSz = KERNEL_TIMING_SAMPLESZ;
static size_t g_timeSampleCt = 0;
static float g_totalKernelTimes = 0.0f;

ParticleData g_particles;
PlaneData g_planes;

static void init() {
    srand(0);

    // CUDA //
        gpuErrchk(cudaSetDevice(0));
        cudaEventCreate(&kernel_simStart);
        cudaEventCreate(&kernel_simStop);

    // Planes //
        const float plane_width = 540.0f;
        g_planes = PlaneData(6, plane_width);
        g_planes.initPlanes();
        g_planes.copyToDevice();

    // Particles //
        g_particles = ParticleData(g_numParticles);
        g_particles.init(0.5f);
        g_particles.copyToDevice();

    size_t problem_sz = g_particles.h_maxParticles;
    g_blocksPerGrid = dim3((problem_sz + g_threadsPerBlock.x - 1) / g_threadsPerBlock.x);
}

/*
    Instead of iterating over each particle, we will make a kernel that runs for each particle
*/

// Assume mass is 1; F / 1 = A
__device__ glm::vec3 getAcceleration(int idx, const glm::vec4* v) {
    float mass = 1.0f;

    // Simple force composed of gravity and air resistance
    glm::vec3 F_total = glm::vec3(GRAVITY) - ((AIR_FRICTION / mass) * glm::vec3(v[idx])); 

    return F_total;
}

/*
    In a flocking simulation, you might have various rules and applications to follow - in this case, we are just
    going to establish a simple distance constraint that is resolve with impulse / momentum.
*/

__device__ void solveConstraints(int idx, const glm::vec4* pos, const glm::vec4* vel, const float* radii,
                                 glm::vec3& x_new, glm::vec3& v_new, float& dt, const glm::vec3& a,
                                 int simulationIdx, int particleIdx, glm::vec3* s_dv) {
    // Truncate the w component
    const glm::vec3& x(pos[idx]), v(vel[idx]);

    // Particle-Particle Collisions //
    /*
        This could be below plane collisions, but seeing as we synchronized threads, we will do it here

        This is the inner loop of for i in range(particles), for j in range(i + 1, particles)

        We can grab the global array value as simulationIdx * particles + j. Note that because we
        are handling j > i particles in the ith thread, the jth thread will never see i - so we should
        update the opposite of the impulse from i -> j to the jth shared velocity; 
    */
    for (int j = particleIdx + 1; j < d_numParticles; j++) {
        int idx_j = simulationIdx * d_numParticles + j;

        glm::vec3 x_j(pos[idx_j]), v_j(vel[idx_j]);
        float r_j = radii[j];

        /*
            If the distance from our particle to the other is less than radii[i] + radii[j] we have collided.

            We can take the direction of j to x and say we (particle i) should be pushed in that direction.

            The extent to which we move, or impulse, is dependent on the relative velocity, or how fast we
            are moving towards each other. For example, if v_rel < 0, we are moving towards each other, and we
            should push off more.

            J = [(1 + e) * v_rel] / [1/m1 + 1/m2]
        */
        glm::vec3 x_ij = x - x_j;
        float d_ij = glm::length(x_ij);

        if (d_ij < radii[idx] + r_j) {
            glm::vec3 n_ij = glm::normalize(x_ij);
            glm::vec3 v_ij = v - v_j;

            float v_rel = glm::dot(v_ij, n_ij);
            float impulse = (1 + RESTITUTION) * v_rel / (1 + 1);

            glm::vec3 impulse_vec = impulse * n_ij;
            s_dv[particleIdx] += impulse_vec;

            // Consequently, we should change the velocities of j. This is not thread safe so we have to atomic
            atomicAdd(&s_dv[j].x, -impulse_vec.x);
            atomicAdd(&s_dv[j].y, -impulse_vec.y);
            atomicAdd(&s_dv[j].z, -impulse_vec.z);
        }
    }

    // Synchronize threads, because we don't want to start plane collisions until this is done
    __syncthreads();
    
    // Plane Collisions //
    for (int i = 0; i < d_numPlanes; i++) {
        const glm::vec3& p(d_planeP[i]), n(d_planeN[i]);

        glm::vec3 new_p = p + (radii[idx] * n);
    }

}

// __device__ void solveConstraints(int idx, const glm::vec3* pos, const glm::vec3* vel, const float* radii, 
//                                  glm::vec3& x_new, glm::vec3& v_new, float& dt, const glm::vec3& a) {
//     // Avoid at rest particles otherwise our counter will be inaccurate
//     if (glm::length(v_new) < STOP_VELOCITY) {
//         return;
//     }

//     // Truncate the w component
//     const glm::vec3& x(pos[idx]), v(vel[idx]);

//     // Plane Collisions //
//     for (int i = 0; i < d_numPlanes; i++) {
//         const glm::vec3& p(d_planeP[i]), n(d_planeN[i]);

//         glm::vec3 new_p = p + (radii[idx] * n);

//         float d_0 = glm::dot(x - new_p, n);
//         float d_n = glm::dot(x_new - new_p, n);

//         glm::vec3 v_tan = v - (glm::dot(v, n) * n);
//         v_tan = (1 - FRICTION) * v_tan;

//         if (d_n < FLOAT_EPS) {
//             float f = d_0 / (d_0 - d_n);
//             dt = f * dt;

//             glm::vec3 v_collision = (v + (dt * a)) * RESTITUTION;    
//             glm::vec3 x_collision = x;

//             x_new = x_collision;
//             v_new = (abs(glm::dot(v_collision, n)) * n) + (v_tan);

//             // Because behavior is pretty standard, naive jitter handling is okay here. Two more checks are possible.
//             if (abs(glm::dot(v_new, n)) < STOP_VELOCITY) {
//                 v_new = v_tan;
//             }
//         }
//     }

//     // If |v_idx| < STOP_VELOCITY we can assume a particle has "converged", and we should reduce the counter.
//     // this works because each particle has a significant nonzero initial velocity.
//     if (glm::length(v_new) < STOP_VELOCITY) {
//         atomicAdd(&d_deadParticles, 1);
//     }
// }

/*
    We are now working with global arrays of great size; 
*/
__global__ void simulateKernel(glm::vec4* pos, glm::vec4* vel, float* radii, int* convergeFlags) {
    unsigned int simulationIdx(blockIdx.x), particleIdx(threadIdx.x);

    // Overflow shouldn't be possible
    int idx = simulationIdx * d_numParticles + particleIdx;

    if (idx > (d_numWorlds * d_numParticles)) {
        return;
    }

    // Allocate shared memory for impulse and convergence - because n varies at runtime
    __shared__ glm::vec3 s_dv[d_numParticles];
    __shared__ int s_converged;

    // We only want to initialize it once
    if (particleIdx == 0) {
        s_converged = 1;
    }

    // Handle fractional timesteps
    float dt_remaining = DT_SIMULATION;
    float dt = dt_remaining;
    short max_iter = 10;
    
    glm::vec3 x_cur(pos[idx]), v_cur(vel[idx]);
    
    while (max_iter && dt_remaining > FLOAT_EPS) {
        // Within the timestep multiple collisions are possible, so we will have to reuse the shared memory 
        s_dv[particleIdx] = glm::vec3(0.0f);
        
        glm::vec3 a = getAcceleration(idx, vel);
        glm::vec3 x_new, v_new;

        // We have to synchronize before and after entering
        __syncthreads();
        
        // TODO: Implement solveConstraints - it should apply necessary changes to velocity and whatnot.
        // solveConstraints(...)

        __syncthreads();

    }

}

// __global__ void simulateKernel(vec3* pos, vec3* vel, float* radii) {
//     /* To retrieve the index in this 1D instance, we do this: */
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
//     // Because we are effectively doing a ceiling function for threads, some amount will not have a particle associated
//     // Additionally, note this is MINUS ONE because we don't want the last one moving.
//     if (idx >= (d_numParticles - 1)) {
//         return;
//     }

//     // Use of __constant__ space on the kernel https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory
//     float dt_remaining = DT_SIMULATION;
//     float dt = dt_remaining;

//     int max_iter = 10;
//     const vec3& x_cur(pos[idx]), v_cur(vel[idx]);

//     while (max_iter && dt_remaining > 0.0f) {
//         // TODO: Consider passing in pos, vel through value to functions like this to avoid uncoalesced global accesses
//         vec3 a = getAcceleration(idx, vel);

//         vec3 x_new, v_new;

//         // Integrate over timestep to update
//         solveConstraints(idx, pos, vel, radii, x_new, v_new, dt, a);

//         // Update particle state
//         pos[idx] = x_new;
//         vel[idx] = v_new;

//         // Update remaining time
//         dt_remaining -= dt;
//         max_iter--;
//     }
// }

long long ctr=10e9;

void launchSimulations() {
    /*
        TODO: The new goal of this function is to iterate over as many 50^2 worlds
        as possible, and branch off a kernel for each of them - this loop should
        be very fast as it 
    */

    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#elapsed-time
    gpuErrchk(cudaEventRecord(kernel_simStart, 0));
    simulateKernel<<<g_blocksPerGrid, g_threadsPerBlock>>>(g_particles.d_position, g_particles.d_velocity, g_particles.d_radii);
    gpuErrchk(cudaEventRecord(kernel_simStop, 0));
    gpuErrchk(cudaEventSynchronize(kernel_simStop));
    
    // TODO: Potentially add an event to avoid the constant memcpy.
    gpuErrchk(cudaMemcpyFromSymbol(&g_deadParticles, d_deadParticles, sizeof(unsigned int), 0, cudaMemcpyDeviceToHost));
 
    float elapsed;
    gpuErrchk(cudaEventElapsedTime(&elapsed, kernel_simStart, kernel_simStop));

    g_totalKernelTimes += elapsed;
    g_timeSampleCt++;

    gpuErrchk(cudaGetLastError());
}

int main(int argc, char**argv) {
    if (argc < 3) {
        cout << "Usage: ./executable <number of particles> <threads per block>" << endl;
        return 0;
    }

    g_numParticles = stoi(argv[1]);
    int threadsPerBlock = stoi(argv[2]);
    g_threadsPerBlock = dim3(threadsPerBlock);

    g_deadParticles = 0;
    gpuErrchk(cudaMemcpyToSymbol(d_deadParticles, &g_deadParticles, sizeof(unsigned int)));

    // Initialize planes, particles, cuda buffers
    init();

    // Program converges when the last moving particle "stops", or the max time is exceeded.
    auto start = std::chrono::high_resolution_clock::now();
    auto end = start + std::chrono::seconds(MAX_SIMULATE_TIME_SECONDS);
    
    while ((std::chrono::high_resolution_clock::now() < end) && (g_deadParticles < g_numParticles)) {
        launchSimulations();
    }

    // Convergence time
    auto conv_time = std::chrono::high_resolution_clock::now() - start;
    auto conv_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(conv_time).count();
    printf("Actual program time: %ld ms\n", conv_time_ms);

    // Print Timings //
    if (BENCHMARK) {
        float overall = g_totalKernelTimes;
        float avg = g_totalKernelTimes / g_timeSampleCt;
        float usage = g_totalKernelTimes / (conv_time_ms);

        printf("Number of threads: %d, number of blocks (per grid): %d\n", g_threadsPerBlock.x, g_blocksPerGrid.x);
        printf("Average simulateKernel() execution time over %d samples: %f ms\n", g_timeSampleCt, avg);
        printf("Overall kernel time before convergence: %f ms\n", overall);
        printf("Kernel time / total program time: %f\n", usage);
    }

    // CUDA Cleanup //
    cudaEventDestroy(kernel_simStart);
    cudaEventDestroy(kernel_simStop);

    return 0;
}
