#include "ParticleData.h"
#include <cuda_runtime.h>

#define _USE_MATH_DEFINES
#include <math.h>
#include <random>
#include "../../include.h"
using std::vector, std::cerr, std::cout, std::endl;

extern __constant__ size_t d_numParticles;

__host__ __device__ ParticleData::ParticleData() : h_numParticles(0), h_numWorlds(0) {
    d_position = nullptr;
    d_velocity = nullptr;
    d_radii = nullptr;
}

__host__ __device__ ParticleData::ParticleData(size_t numParticles, size_t numWorlds) : h_numParticles(numParticles), h_numWorlds(numWorlds) {
    int k(numWorlds), n(numParticles);
    size_t mem_required = (k * n) * (2 * sizeof(glm::vec4) + sizeof(float));

    size_t free, total;
    cudaMemGetInfo(&free, &total);

    if (mem_required > free) {
        cerr << "FATAL: " << __FILE__ << ": " << __LINE__ << endl;
        cerr << "FATAL: Insufficient memory to cudaMalloc " << mem_required << " bytes on device for particles " << endl;
        exit(1);
    }
    else {
        cout << "Allocating " << mem_required << " of " << total << " bytes on device" << endl;
    }

    gpuErrchk(cudaMalloc(&d_position, k * n * sizeof(glm::vec4)));
    gpuErrchk(cudaMalloc(&d_velocity, k * n * sizeof(glm::vec4)));
    gpuErrchk(cudaMalloc(&d_radii, k * n * sizeof(float)));
    gpuErrchk(cudaMalloc(&d_convergenceFlags, k * sizeof(int)));
}

// FIXME: Causes issues if I (attempt to) properly handle memory deallocation.
__host__ __device__ ParticleData::~ParticleData() {
    // if (d_position) {
    //     cout << "cudaFree'ing d_position* = " << d_position << endl; 
    //     gpuErrchk(cudaFree(&d_position));
    // }
    // if (d_velocity) {
    //     gpuErrchk(cudaFree(&d_velocity));
    // }
    // if (d_radii) {
    //     gpuErrchk(cudaFree(&d_radii));
    // }

    // printf("Destructor called %d time(s) for this=%p\n", ++destructorCt, this);
}

// Copies physics data from host to device (this should happen only once!)
__host__ __device__ void ParticleData::copyToDevice() {
    size_t k(h_numWorlds), n(h_numParticles);
    size_t instances = k * n;

    gpuErrchk(cudaMemcpyToSymbol(&d_numParticles, &n, sizeof(size_t)));

    gpuErrchk(cudaMemcpy(d_position, h_position.data(), instances * sizeof(glm::vec4), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_velocity, h_velocity.data(), instances * sizeof(glm::vec4), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_radii, h_radii.data(), instances * sizeof(float), cudaMemcpyHostToDevice));
}

// Arbitrary population of on-host memory for rendering with vsh & fsh
void ParticleData::init(const float& radius) {
    size_t k(h_numWorlds), n(h_numParticles);
    assert(k > 0);
    assert(n > 0);

    h_position.resize(k * n);
    h_velocity.resize(k * n);
    h_radii.resize(k * n);

    // Randomization //
        std::random_device rd;
        // std::mt19937 gen(rd());
        std::mt19937 gen(0);
        float minF(180.0f), maxF(240.0f);
        std::uniform_real_distribution<float> dist_pos(minF, maxF);
        std::uniform_real_distribution<float> dist_vel(-1.0f, 1.0f);
    
    // For each loops with & slightly slower, but readable + we are in initialization stage
    for (glm::vec4& x : h_position) {
        float rand_x(dist_pos(gen));
        float rand_y(dist_pos(gen));
        float rand_z(dist_pos(gen));

        x = glm::vec4(rand_x - maxF, rand_y + minF, rand_z - maxF, 0.0f);
        // cout << x.x << " " << x.y << " " << x.z << endl;
    }

    float push = 10.0f;
    for (glm::vec4& v : h_velocity) {
        float rand_x = dist_vel(gen);
        float rand_y = dist_vel(gen);
        float rand_z = dist_vel(gen);

        v = glm::vec4(push * rand_x, push * rand_y, push * rand_z, 0.0f);
    }

    for (float& r : h_radii) {
        r = radius;
    }

    assert(h_position.size() == k * n);
    assert(h_position.size() == h_velocity.size());
    assert(h_position.size() == h_radii.size());
}