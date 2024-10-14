#pragma once
#ifndef PARTICLEDATA_H
#define PARTICLEDATA_H

#include <glm/glm.hpp>
#include <vector>
#include <cuda_runtime.h>

using std::vector;

/*
    Instead of a particle class, we can store all the arrays of information in one struct for readability - that way
    unnecessary data isn't passed in. We will store this information on the device, and we can populate it on the host.

    Note position is not a pointer here, that is because we will store it in a VBO for OpenGL rendering.

    Store any data for rendering on host (or stuff to send to GPU once), and any data for physics on device.
*/
class ParticleData {
    public:
        vector<glm::vec4> h_position;
        vector<glm::vec4> h_velocity;
        vector<float> h_radii;

        glm::vec4* d_position;
        glm::vec4* d_velocity;
        float* d_radii;

        size_t h_numParticles;
        size_t h_numWorlds;
           
        __host__ __device__ ParticleData();
        __host__ __device__ ParticleData(size_t numParticles, size_t numWorlds);
        __host__ __device__ ~ParticleData();

        void __host__ __device__ copyToDevice();
        void __host__ __device__ init(const float& radius=0.5f);

        int destructorCt = 0;
};


#endif // PARTICLEDATA_H
